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

    def test_distinct_odd_values(self):
        # a Decimal is no Cypher number, and comparing it with a NumPy integer raises
        from decimal import Decimal

        import numpy as np
        g = ec.Graph().add_nodes([
            ("a", {"v": Decimal(1)}), ("b", {"v": np.int64(1)}), ("c", {"v": 1}),
            ("d", {"v": Decimal("1.0")}), ("e", {"v": {1, 2}}), ("f", {"v": {1, 2}}),
        ])
        assert g.match("MATCH (n) RETURN DISTINCT n.v").height == 3

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


class TestReturnExpressions:
    def test_aliases_and_names(self, g):
        df = g.match(ACTED + " RETURN p.name AS name, toUpper(m.title), m AS movie")
        assert df.columns == ["name", "toUpper(m.title)", "movie"]
        assert df.dtypes == [pl.String, pl.String, pl.Int64]
        assert sorted(df.rows()[i][:2] for i in range(3)) == [
            ("Carrie-Anne Moss", "THE MATRIX"), ("Keanu Reeves", "JOHN WICK"),
            ("Keanu Reeves", "THE MATRIX")]

    def test_dtypes(self, g):
        df = g.match(ACTED + " RETURN id(p), type(r), labels(m), m.released > 2000 AS new, "
                     "size(r.roles), p.born IS NULL, [p.name, m.title] AS names")
        assert df.dtypes == [pl.Int64, pl.String, pl.List(pl.String), pl.Boolean, pl.Int64,
                             pl.Boolean, pl.List(pl.String)]
        assert set(df["type(r)"]) == {"ACTED_IN"}
        assert df["labels(m)"].to_list() == [["Movie"]] * 3

    def test_no_matches_keeps_the_dtypes(self, g):
        df = g.match(ACTED + " WHERE false RETURN id(p) AS i, type(r) AS t, p.name")
        assert df.height == 0 and df.dtypes == [pl.Int64, pl.String, pl.Null]

    def test_path_functions(self, g):
        df = g.match("MATCH p = shortestPath((a {name: 'Carrie-Anne Moss'})-[*]-(b {name: "
                     "'wick'})) RETURN length(p) AS hops, nodes(p), relationships(p)")
        assert df.columns == ["hops", "nodes(p)", "relationships(p)"]
        assert df.dtypes == [pl.Int64, pl.List(pl.Int64), pl.List(pl.Int64)]
        assert df.row(0) == (3, [2, 4, 1, 5], [2, 1, 3])
        df = g.match("MATCH p = shortestPath((a {name: 'Carrie-Anne Moss'})-[*]-(b {name: "
                     "'wick'})) RETURN nodes(p) AS ns", use_labels=True)
        assert df.dtypes == [pl.List(pl.String)]
        assert df["ns"].to_list() == [["Carrie-Anne Moss", "matrix", "Keanu Reeves", "wick"]]

    def test_use_labels_inside_values(self, g):
        df = g.match(ACTED + " RETURN [p, m] AS pair, id(p) AS i ORDER BY i, m",
                     use_labels=True)
        assert df.rows() == [
            (["Keanu Reeves", "matrix"], 1), (["Keanu Reeves", "wick"], 1),
            (["Carrie-Anne Moss", "matrix"], 2)]

    def test_star(self, g):
        df = g.match(ACTED + " RETURN *, p.name AS name")
        assert df.columns == ["p", "r", "m", "name"]
        assert df.drop("name").equals(g.match(ACTED))

    def test_value_errors(self, g):
        with pytest.raises(ValueError, match=r"^Cypher: toLower\(\) .*line 1, column"):
            g.match("MATCH (p:Person) RETURN toLower(p.born)")
        # A column that's never computed raises nothing
        assert g.match("MATCH (p:Person) RETURN toLower(p.born) LIMIT 0").height == 0

    def test_errors_are_raised_on_an_empty_graph(self):
        with pytest.raises(ValueError, match="the variable `b` isn't defined"):
            ec.Graph().match("MATCH (a) RETURN a ORDER BY b")
        with pytest.raises(ValueError, match="toLower"):
            ec.Graph().match("MATCH (a) RETURN toLower()")
        with pytest.raises(ValueError, match="LIMIT needs a non-negative integer"):
            ec.Graph().match("MATCH (a) RETURN a LIMIT -1")


class TestOrderBy:
    def test_ascending_and_descending(self, g):
        df = g.match("MATCH (p:Person) RETURN p.name ORDER BY p.born")
        assert df["p.name"].to_list() == ["Keanu Reeves", "Lana Wachowski", "Carrie-Anne Moss"]
        df = g.match("MATCH (p:Person) RETURN p.name ORDER BY p.born DESC")
        assert df["p.name"].to_list() == ["Carrie-Anne Moss", "Lana Wachowski", "Keanu Reeves"]

    def test_by_a_column_name(self, g):
        df = g.match("MATCH (p:Person) RETURN p.born AS name ORDER BY name DESC")
        assert df["name"].to_list() == [1967, 1965, 1964]

    def test_by_an_expression_that_isnt_returned(self, g):
        df = g.match("MATCH (p:Person) RETURN p ORDER BY size(p.name), p.name", use_labels=True)
        assert df["p"].to_list() == ["Keanu Reeves", "Lana Wachowski", "Carrie-Anne Moss"]

    def test_several_keys_and_ties(self, g):
        df = g.match(ACTED + " RETURN p.name, m.title ORDER BY p.name DESC, m.released DESC")
        assert df.rows() == [("Keanu Reeves", "John Wick"), ("Keanu Reeves", "The Matrix"),
                             ("Carrie-Anne Moss", "The Matrix")]
        # Ties keep the order of the rows without ORDER BY
        df = g.match(ACTED + " RETURN r ORDER BY m.title")
        assert df["r"].to_list() == [3, 1, 2]

    def test_nulls_last_ascending_and_first_descending(self, g):
        df = g.match("MATCH (n) RETURN n ORDER BY n.born", use_labels=True)
        assert df["n"].to_list() == [
            "Keanu Reeves", "Lana Wachowski", "Carrie-Anne Moss", "matrix", "wick"]
        df = g.match("MATCH (n) RETURN n ORDER BY n.born DESC", use_labels=True)
        assert df["n"].to_list() == [
            "matrix", "wick", "Carrie-Anne Moss", "Lana Wachowski", "Keanu Reeves"]

    def test_mixed_types(self):
        values = [3, "b", None, [1], True, 1.5, "a", False, float("nan"), [0, 2]]
        g = ec.Graph().add_nodes([(f"n{i}", {"v": v}) for i, v in enumerate(values)])
        df = g.match("MATCH (n) RETURN n ORDER BY n.v", use_labels=True)
        # lists < strings < booleans < numbers (NaN last) < null
        assert df["n"].to_list() == [
            "n9", "n3", "n6", "n1", "n7", "n4", "n5", "n0", "n8", "n2"]

    def test_numbers_compare_across_int_and_float(self):
        g = ec.Graph().add_nodes([("a", {"v": 2}), ("b", {"v": 1.5}), ("c", {"v": -1})])
        assert g.match("MATCH (n) RETURN n ORDER BY n.v", use_labels=True)["n"].to_list() == [
            "c", "b", "a"]

    def test_after_distinct(self, g):
        df = g.match(ACTED + " RETURN DISTINCT m.title AS t ORDER BY t")
        assert df["t"].to_list() == ["John Wick", "The Matrix"]
        df = g.match(ACTED + " RETURN DISTINCT m ORDER BY m.released DESC", use_labels=True)
        assert df["m"].to_list() == ["wick", "matrix"]

    def test_by_nodes_and_relationships(self, g):
        df = g.match(ACTED + " RETURN m.title ORDER BY r DESC")
        assert df["m.title"].to_list() == ["John Wick", "The Matrix", "The Matrix"]


class TestSkipLimit:
    def test_skip_and_limit(self, g):
        text = "MATCH (p:Person) RETURN p.name ORDER BY p.born"
        assert g.match(text + " SKIP 1")["p.name"].to_list() == [
            "Lana Wachowski", "Carrie-Anne Moss"]
        assert g.match(text + " LIMIT 2")["p.name"].to_list() == [
            "Keanu Reeves", "Lana Wachowski"]
        assert g.match(text + " SKIP 1 LIMIT 1")["p.name"].to_list() == ["Lana Wachowski"]
        assert g.match(text + " SKIP 5").height == 0
        assert g.match(text + " LIMIT 0").height == 0

    def test_without_order_by(self, g):
        assert g.match(ACTED + " RETURN r SKIP 1")["r"].to_list() == [3, 2]
        assert g.match(ACTED + " RETURN r LIMIT 2")["r"].to_list() == [1, 3]

    def test_params(self, g):
        df = g.match("MATCH (p:Person) RETURN p.name ORDER BY p.born SKIP $s LIMIT $n",
                     params={"s": 1, "n": 1})
        assert df["p.name"].to_list() == ["Lana Wachowski"]

    @pytest.mark.parametrize("value", ["-1", "1.5", "true", "'2'", "null", "[1]", "$v"])
    def test_bad_values(self, g, value):
        for clause in ("SKIP", "LIMIT"):
            with pytest.raises(ValueError, match=rf"^Cypher: {clause} needs a non-negative "
                                                 r"integer, not .*line 1, column"):
                g.match(f"{ACTED} RETURN p {clause} {value}", params={"v": -2})
            with pytest.raises(ValueError, match=rf"{clause} needs a non-negative integer"):
                g.count_matches(f"{ACTED} RETURN p {clause} {value}", params={"v": -2})

    def test_with_the_limit_argument(self, g):
        # `limit=` caps the matches before ORDER BY, SKIP and LIMIT
        text = "MATCH (p:Person) RETURN p.name ORDER BY p.born DESC"
        assert g.match(text, limit=2)["p.name"].to_list() == ["Carrie-Anne Moss", "Keanu Reeves"]
        assert g.match(text + " SKIP 1", limit=2)["p.name"].to_list() == ["Keanu Reeves"]
        assert g.match(text + " LIMIT 5", limit=1).height == 1

    def test_distinct(self, g):
        df = g.match(ACTED + " RETURN DISTINCT p.name ORDER BY p.name SKIP 1")
        assert df["p.name"].to_list() == ["Keanu Reeves"]


class TestAggregation:
    def test_count_per_group(self, g):
        df = g.match(ACTED + " RETURN m.title AS title, count(*) AS n ORDER BY n DESC")
        assert df.rows() == [("The Matrix", 2), ("John Wick", 1)]
        assert df.dtypes == [pl.String, pl.Int64]

    def test_groups_keep_the_place_of_their_first_row(self, g):
        # rows sort by (p, r, m), so Keanu's group comes first, and collect() keeps row order
        df = g.match(ACTED + " RETURN p.name, collect(m.title) AS movies")
        assert df.rows() == [("Keanu Reeves", ["The Matrix", "John Wick"]),
                             ("Carrie-Anne Moss", ["The Matrix"])]
        assert g.match(ACTED + " RETURN m.title, count(*)").rows() == [
            ("The Matrix", 2), ("John Wick", 1)]

    def test_without_grouping_keys(self, g):
        df = g.match("MATCH (p:Person) RETURN count(*) AS n, min(p.born), max(p.born), "
                     "sum(p.born), avg(p.born), collect(p.name)")
        assert df.rows() == [(3, 1964, 1967, 5896, 5896 / 3,
                              ["Keanu Reeves", "Carrie-Anne Moss", "Lana Wachowski"])]

    def test_no_matches(self, g):
        # Without grouping keys there's still one row; with them, there are no groups
        df = g.match("MATCH (p:Robot) RETURN count(*), count(p), sum(p.born), avg(p.born), "
                     "min(p.born), collect(p.name)")
        assert df.rows() == [(0, 0, 0, None, None, [])]
        assert df.dtypes[:4] == [pl.Int64, pl.Int64, pl.Int64, pl.Float64]
        df = g.match("MATCH (p:Robot) RETURN p.name, count(*) AS n")
        assert df.height == 0 and df.columns == ["p.name", "n"]
        assert g.match("MATCH (p:Robot) RETURN count(*)", limit=0).rows() == [(0,)]

    def test_nulls_are_skipped(self, g):
        # only the people have `born`
        df = g.match("MATCH (n) RETURN count(*), count(n.born), collect(n.born), min(n.born), "
                     "avg(n.born)")
        assert df.rows() == [(5, 3, [1964, 1967, 1965], 1964, 5896 / 3)]

    def test_distinct(self, g):
        df = g.match(ACTED + " RETURN count(DISTINCT p) AS people, count(p) AS credits, "
                     "collect(DISTINCT p.name) AS names")
        assert df.rows() == [(2, 3, ["Keanu Reeves", "Carrie-Anne Moss"])]
        g2 = ec.Graph().add_nodes([("a", {"x": 1}), ("b", {"x": 1}), ("c", {"x": 2.0}),
                                   ("d", {"x": 2})])
        assert g2.match("MATCH (n) RETURN sum(DISTINCT n.x), count(DISTINCT n.x), "
                        "collect(DISTINCT n.x)").rows() == [(3.0, 2, [1.0, 2.0])]

    def test_group_by_nodes(self, g):
        df = g.match(ACTED + " RETURN p, count(*) AS n, collect(m) AS movies, min(m) AS first")
        assert df.rows() == [(1, 2, [4, 5], 4), (2, 1, [4], 4)]
        assert df.dtypes == [pl.Int64, pl.Int64, pl.List(pl.Int64), pl.Int64]
        df = g.match(ACTED + " RETURN p, count(*) AS n, collect(m) AS movies, min(m) AS first",
                     use_labels=True)
        assert df.rows() == [("Keanu Reeves", 2, ["matrix", "wick"], "matrix"),
                             ("Carrie-Anne Moss", 1, ["matrix"], "matrix")]
        assert df.dtypes == [pl.String, pl.Int64, pl.List(pl.String), pl.String]

    def test_dtypes(self, g):
        df = g.match(ACTED + " RETURN count(*), avg(p.born), collect(r), collect(m.title), "
                     "collect(type(r)), sum(id(p)), max(m.title), min(m.released > 2000)")
        assert df.dtypes == [pl.Int64, pl.Float64, pl.List(pl.Int64), pl.List(pl.String),
                             pl.List(pl.String), pl.Int64, pl.String, pl.Boolean]
        assert df.row(0) == (3, (1964 * 2 + 1967) / 3, [1, 3, 2],
                             ["The Matrix", "John Wick", "The Matrix"],
                             ["ACTED_IN"] * 3, 4, "The Matrix", False)

    def test_expressions_of_aggregations_and_keys(self, g):
        df = g.match(ACTED + " RETURN m.title, [m.title, count(*)] AS pair, "
                     "size(collect(p)) AS n, toLower(max(p.name)) AS last")
        assert df.rows() == [("The Matrix", ["The Matrix", 2], 2, "keanu reeves"),
                             ("John Wick", ["John Wick", 1], 1, "keanu reeves")]

    def test_order_skip_and_limit(self, g):
        text = "MATCH (p:Person)-->(m) RETURN m.title AS t, count(*) AS n"
        assert g.match(text + " ORDER BY n DESC, t").rows() == [
            ("The Matrix", 3), ("John Wick", 1)]
        assert g.match(text + " ORDER BY count(*), t DESC").rows() == [
            ("John Wick", 1), ("The Matrix", 3)]
        assert g.match(text + " ORDER BY n DESC SKIP 1").rows() == [("John Wick", 1)]
        assert g.match(text + " ORDER BY n DESC LIMIT 1").rows() == [("The Matrix", 3)]
        # by an expression of the grouping keys that isn't returned
        assert g.match(ACTED + " RETURN m, count(*) AS n ORDER BY m.released DESC").rows() == [
            (5, 1), (4, 2)]

    def test_return_distinct(self, g):
        df = g.match(ACTED + " RETURN DISTINCT m.title, count(*) AS n ORDER BY n")
        assert df.rows() == [("John Wick", 1), ("The Matrix", 2)]

    def test_min_and_max_follow_order_by(self):
        g = ec.Graph().add_nodes([("a", {"v": 1}), ("b", {"v": "z"}), ("c", {"v": True}),
                                  ("d", {"v": [1]}), ("e", {"v": 2.5}), ("f", {})])
        # lists sort before strings, strings before booleans, and booleans before numbers
        assert g.match("MATCH (n) RETURN min(n.v), max(n.v)").rows() == [([1], 2.5)]
        df = g.match("MATCH (n) WHERE n.v IN ['z', 1, 2.5] RETURN min(n.v) AS lo, max(n.v) AS hi")
        assert df.rows() == [("z", 2.5)]

    def test_sum_and_avg(self):
        g = ec.Graph().add_nodes([("a", {"v": 1}), ("b", {"v": 2}), ("c", {"v": 0.5}), ("d", {})])
        assert g.match("MATCH (n) WHERE n.v <> 0.5 RETURN sum(n.v), avg(n.v)").rows() == [
            (3, 1.5)]
        assert g.match("MATCH (n) RETURN sum(n.v), avg(n.v)").rows() == [(3.5, 3.5 / 3)]

    def test_sum_and_avg_of_other_number_types(self):
        # numpy integers add as Python ints, so they don't wrap around or overflow
        np = pytest.importorskip("numpy")
        from fractions import Fraction

        g = ec.Graph().add_nodes([("a", {"v": np.int64(2**62)}), ("b", {"v": np.int64(2**62)}),
                                  ("c", {"v": 2**70})])
        total = g.match("MATCH (n) RETURN sum(n.v) AS s").item()
        assert total == 2**63 + 2**70 and type(total) is int
        g = ec.Graph().add_nodes([("a", {"v": Fraction(1, 2)}), ("b", {"v": np.float32(0.5)})])
        df = g.match("MATCH (n) RETURN sum(n.v) AS s, avg(n.v) AS a")
        assert df.rows() == [(1.0, 0.5)] and df["a"].dtype == pl.Float64

    def test_nan_values_group_together(self):
        g = ec.Graph().add_nodes([("a", {"v": float("nan")}), ("b", {"v": float("nan")}),
                                  ("c", {"v": 1})])
        df = g.match("MATCH (n) RETURN n.v, count(*) AS k")
        assert df["k"].to_list() == [2, 1]
        assert g.match("MATCH (n) RETURN count(DISTINCT n.v)").item() == 2
        assert g.match("MATCH (n) RETURN DISTINCT n.v").height == 2

    def test_limit_argument_caps_the_matches_aggregated(self, g):
        assert g.match("MATCH (p:Person) RETURN count(*)", limit=2).item() == 2

    def test_star_and_aggregation(self, g):
        df = g.match("MATCH (p)-[:ACTED_IN]->(m) RETURN *, count(*) AS n")
        assert df.columns == ["p", "m", "n"] and df["n"].to_list() == [1, 1, 1]

    def test_value_errors(self, g):
        with pytest.raises(ValueError, match=r"^Cypher: sum\(\) needs numbers, not 'Keanu Reeves' "
                                             r"\(line 1, column 33\)"):
            g.match("MATCH (p:Person) RETURN p.born, sum(p.name)")
        with pytest.raises(ValueError, match=r"avg\(\) needs numbers, not a node"):
            g.match("MATCH (p:Person) RETURN avg(p)")
        with pytest.raises(ValueError, match=r"avg\(\) needs numbers, not True"):
            g.match("MATCH (p:Person) RETURN avg(p.born > 0)")

    def test_query_errors(self, g):
        with pytest.raises(ValueError, match=r"`m` is used beside an aggregation"):
            g.match(ACTED + " RETURN p.name, [m.title, count(*)]")
        with pytest.raises(ValueError, match=r"after an aggregation, ORDER BY can only use"):
            g.match(ACTED + " RETURN p.name, count(*) ORDER BY m.title")
        with pytest.raises(ValueError, match=r"count\(\) is an aggregating function, which only"):
            g.match(ACTED + " WHERE count(*) > 1 RETURN p")
        with pytest.raises(ValueError, match=r"select_match\(\) doesn't take RETURN"):
            g.select_match(ACTED + " RETURN count(*)")


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

    def test_skip_and_limit(self, g):
        assert g.count_matches(ACTED + " RETURN p SKIP 1") == 2
        assert g.count_matches(ACTED + " RETURN p SKIP 5") == 0
        assert g.count_matches(ACTED + " RETURN p LIMIT 2") == 2
        assert g.count_matches(ACTED + " RETURN p ORDER BY p.name SKIP 1 LIMIT 5") == 2
        assert g.count_matches(ACTED + " RETURN DISTINCT p SKIP 1") == 1
        assert g.count_matches(ACTED + " RETURN p LIMIT $n", {"n": 1}) == 1

    def test_counts_rows_of_match_with_expressions(self, g):
        for text in ("RETURN toUpper(p.name) AS n ORDER BY n", "RETURN DISTINCT size(r.roles)",
                     "RETURN DISTINCT p.born > 1965 ORDER BY p.born > 1965 SKIP 1",
                     "RETURN * LIMIT 1"):
            assert g.count_matches(ACTED + " " + text) == g.match(ACTED + " " + text).height

    def test_columns_are_only_computed_for_distinct(self, g):
        assert g.count_matches("MATCH (p:Person) RETURN toLower(p.born)") == 3
        with pytest.raises(ValueError, match="toLower"):
            g.count_matches("MATCH (p:Person) RETURN DISTINCT toLower(p.born)")

    def test_aggregation_counts_groups(self, g):
        assert g.count_matches(ACTED + " RETURN m.title, count(*)") == 2
        assert g.count_matches(ACTED + " RETURN p, m.title, collect(r)") == 3
        assert g.count_matches(ACTED + " RETURN count(*)") == 1
        assert g.count_matches("MATCH (p:Robot) RETURN count(*)") == 1
        assert g.count_matches("MATCH (p:Robot) RETURN p.name, count(*)") == 0
        assert g.count_matches(ACTED + " RETURN m.title, count(*) SKIP 1") == 1
        assert g.count_matches(ACTED + " RETURN count(*) SKIP 1") == 0

    def test_aggregations_arent_computed(self, g):
        text = "MATCH (p:Person) RETURN p.born, sum(p.name)"
        assert g.count_matches(text) == 3
        with pytest.raises(ValueError, match="sum"):
            g.match(text)

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
# Variable-length relationships and several clauses
# =============================================================================


@pytest.fixture
def chain():
    return ec.Graph(directed=True).add_nodes(list("ABCD")).add_edges(
        [("A", "B"), ("B", "C"), ("C", "D")])


def edge_ids(g):
    return sorted(g.get_edge_ids(limit=None))


class TestPaths:
    def test_a_path_is_a_list_column(self, chain):
        e1, e2, e3 = edge_ids(chain)
        df = chain.match("MATCH (a {name: 'A'})-[r*1..3]->(b)")
        assert df.schema == {"a": pl.Int64, "r": pl.List(pl.Int64), "b": pl.Int64}
        assert df["r"].to_list() == [[e1], [e1, e2], [e1, e2, e3]]
        assert labels(chain, df["b"]) == ["B", "C", "D"]

    def test_paths_in_return_and_distinct(self, chain):
        df = chain.match("MATCH (a)-[r*0..1]->(b) WHERE size(r) = 0 RETURN DISTINCT r")
        assert df.schema == {"r": pl.List(pl.Int64)} and df["r"].to_list() == [[]]
        assert chain.count_matches("MATCH (a)-[r*0..1]->(b) RETURN DISTINCT r") == 4

    def test_count(self, chain):
        assert chain.count_matches("MATCH (a)-[*]->(b)") == 6
        assert chain.count_matches("MATCH (a)-[*]-(b)") == 12
        assert chain.count_matches("MATCH (a)-[*0..]-(b)") == 16

    def test_select_the_edges_of_each_path(self, chain):
        chain.select_match("MATCH (a {name: 'A'})-[r*2]->(b)", select="r")
        assert chain.count_selected_edges() == 2
        chain.select_match("MATCH (a {name: 'B'})-[r*]->(b), (b)-[s]->()", select=["r", "s"],
                           set_op="replace")
        assert sorted(chain.get_selected_edge_ids()) == edge_ids(chain)[1:]

    def test_selecting_nodes_takes_the_inner_ones_too(self, chain):
        chain.select_match("MATCH (a {name: 'A'})-[*3]->(b)")
        assert labels(chain, chain.get_selected_node_ids()) == ["A", "B", "C", "D"]
        chain.select_match("MATCH (a {name: 'A'})-[*3]->(b)", select="b", set_op="replace")
        assert labels(chain, chain.get_selected_node_ids()) == ["D"]

    def test_kinds_still_dont_mix(self, chain):
        with pytest.raises(ValueError, match="both node and relationship variables"):
            chain.select_match("MATCH (a)-[r*]->(b)", select=["a", "r"])

    def test_step_highlights_the_whole_path(self, chain):
        chain.count_matches("MATCH (a {name: 'A'})-[*3]->(b)")
        delta = chain.steps[-1].delta
        assert delta.nodes_selected == ["A", "D", "B", "C"]
        assert delta.edges_selected == [("A", "B"), ("B", "C"), ("C", "D")]


class TestPathVariables:
    def test_a_path_is_a_struct_column(self, chain):
        e1, e2, _ = edge_ids(chain)
        a, b, c = (node(chain, x) for x in "ABC")
        df = chain.match("MATCH p = (x {name: 'A'})-[*1..2]->(y)")
        assert df.schema == {
            "p": pl.Struct({"nodes": pl.List(pl.Int64), "relationships": pl.List(pl.Int64)}),
            "x": pl.Int64, "y": pl.Int64}
        assert df["p"].to_list() == [{"nodes": [a, b], "relationships": [e1]},
                                     {"nodes": [a, b, c], "relationships": [e1, e2]}]
        # the nodes and relationships unnest into list columns
        assert df.select(pl.col("p").struct.field("nodes"))["nodes"].to_list() == [
            [a, b], [a, b, c]]

    def test_use_labels(self, chain):
        df = chain.match("MATCH p = (x {name: 'A'})-[*1..2]->(y) RETURN p", use_labels=True)
        assert df.schema["p"] == pl.Struct({"nodes": pl.List(pl.String),
                                            "relationships": pl.List(pl.Int64)})
        assert [row["nodes"] for row in df["p"]] == [["A", "B"], ["A", "B", "C"]]

    def test_distinct_and_order(self, chain):
        df = chain.match("MATCH p = (x)-[*0..1]->(y) RETURN DISTINCT p", use_labels=True)
        assert [row["nodes"] for row in df["p"]] == [
            ["A"], ["A", "B"], ["B"], ["B", "C"], ["C"], ["C", "D"], ["D"]]
        assert chain.count_matches("MATCH p = (x)-[*0..1]-(y) RETURN DISTINCT p") == 10
        assert chain.count_matches("MATCH p = (x)-[*0..1]-(y) RETURN DISTINCT x") == 4

    def test_null_paths(self, chain):
        df = chain.match("MATCH (x {name: 'D'}) OPTIONAL MATCH p = (x)-->(y) RETURN x, p",
                         use_labels=True)
        assert df.rows() == [("D", None)]
        assert df.schema["p"] == pl.Struct({"nodes": pl.List(pl.String),
                                            "relationships": pl.List(pl.Int64)})

    def test_no_matches_keeps_the_type(self, chain):
        df = chain.match("MATCH p = (x)-->(x)")
        assert df.height == 0 and isinstance(df.schema["p"], pl.Struct)

    def test_shortest_paths(self):
        g = ec.Graph(directed=True).add_nodes(list("ABCD")).add_edges(
            [("A", "B"), ("B", "D"), ("A", "C"), ("C", "D"), ("A", "D"), ("D", "A")])
        df = g.match("MATCH p = allShortestPaths((x {name: 'A'})-[*]->(y {name: 'D'})) "
                     "WHERE length(p) > 1",
                     use_labels=True)
        assert [row["nodes"] for row in df["p"]] == [["A", "B", "D"], ["A", "C", "D"]]
        df = g.match("MATCH p = shortestPath((x {name: 'B'})-[*]->(y {name: 'C'})) RETURN p",
                     use_labels=True)
        assert [row["nodes"] for row in df["p"]] == [["B", "D", "A", "C"]]
        assert g.count_matches("MATCH shortestPath((x)-[*]->(y)) WHERE x <> y") == 12

    def test_select_the_nodes_of_each_path(self, chain):
        chain.select_match("MATCH p = (x {name: 'B'})-[*]->(y {name: 'D'})", select="p")
        assert labels(chain, chain.get_selected_node_ids()) == ["B", "C", "D"]
        assert chain.count_selected_edges() == 0
        chain.select_match("MATCH p = (x {name: 'A'})-->(y)", select=["p", "x"],
                           set_op="replace")
        assert labels(chain, chain.get_selected_node_ids()) == ["A", "B"]
        chain.select_match("MATCH (x {name: 'D'}) OPTIONAL MATCH p = (x)-->()", select="p",
                           set_op="replace")
        assert chain.count_selected_nodes() == 0

    def test_a_path_is_a_node_variable_for_select(self, chain):
        with pytest.raises(ValueError, match="both node and relationship variables"):
            chain.select_match("MATCH p = (a)-[r]->(b)", select=["p", "r"])

    def test_path_properties_are_an_error(self, chain):
        with pytest.raises(ValueError, match=r"`p` is a path, so it has no properties"):
            chain.match("MATCH p = (a)-->(b) RETURN p.name")


DIRECTORS = "MATCH (p:Person) OPTIONAL MATCH (p)-[r:DIRECTED]->(m)"


class TestClauses:
    def test_optional_columns_hold_nulls(self, g):
        df = g.match(DIRECTORS + " RETURN p.name, r, m.title")
        assert df.dtypes == [pl.String, pl.Int64, pl.String]
        directed = g.match("MATCH ()-[r:DIRECTED]->()")["r"][0]
        assert df.rows() == [("Keanu Reeves", None, None), ("Carrie-Anne Moss", None, None),
                             ("Lana Wachowski", directed, "The Matrix")]

    def test_use_labels_keeps_nulls(self, g):
        df = g.match(DIRECTORS, use_labels=True)
        assert df.dtypes == [pl.String, pl.Int64, pl.String]
        assert df["m"].to_list() == [None, None, "matrix"]

    def test_null_rows_sort_on_the_other_columns(self, g):
        df = g.match("OPTIONAL MATCH (x:Nope) MATCH (p:Person)", use_labels=True)
        assert df.rows() == [(None, "Keanu Reeves"), (None, "Carrie-Anne Moss"),
                             (None, "Lana Wachowski")]

    def test_count_includes_null_rows(self, g):
        assert g.count_matches(DIRECTORS) == 3
        assert g.count_matches("MATCH (p:Person) MATCH (p)-[:DIRECTED]->(m)") == 1
        assert g.count_matches("OPTIONAL MATCH (x:Nope)") == 1

    def test_select_skips_nulls(self, g):
        g.select_match(DIRECTORS, select="m")
        assert labels(g, g.get_selected_node_ids()) == ["matrix"]
        g.select_match(DIRECTORS, set_op="replace")
        assert labels(g, g.get_selected_node_ids()) == [
            "Carrie-Anne Moss", "Keanu Reeves", "Lana Wachowski", "matrix"]
        g.select_match("OPTIONAL MATCH (x:Nope)", set_op="replace")
        assert g.count_selected_nodes() == 0

    def test_later_clauses_see_earlier_variables(self, g):
        df = g.match("MATCH (m:Movie {title: 'The Matrix'}) MATCH (p)-[:ACTED_IN]->(m) "
                     "WHERE p.born > 1965 RETURN p.name")
        assert df["p.name"].to_list() == ["Carrie-Anne Moss"]


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
