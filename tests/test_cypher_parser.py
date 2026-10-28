"""Tests for the Cypher parser (`edgecase.cypher`): tokens, patterns, expressions and errors."""

import pytest
from edgecase.cypher import (
    BoolOp,
    Call,
    Compare,
    HasLabels,
    IsNull,
    ListExpr,
    Literal,
    Neg,
    NodePattern,
    Not,
    Param,
    Prop,
    RelPattern,
    Var,
    parse_create,
    parse_query,
    walk,
)


def _where(condition):
    """The WHERE expression of a query over `(a)-[r]->(b)`."""
    return parse_query(f"MATCH (a)-[r]->(b) WHERE {condition}").where


# =============================================================================
# Patterns
# =============================================================================


class TestPatterns:
    def test_nodes(self):
        q = parse_query("MATCH (p:Person:Actor {name: 'Tom', born: $year}), (:Movie|Show), ()")
        p, movie, empty = (pattern.nodes[0] for pattern in q.patterns)
        assert p == NodePattern("p", [("Person",), ("Actor",)],
                                {"name": Literal("Tom"), "born": Param("year")})
        assert movie == NodePattern(None, [("Movie", "Show")])
        assert empty == NodePattern()
        assert q.variables == {"p": "node"}

    def test_relationships(self):
        q = parse_query("MATCH (a)-[r:KNOWS|LIKES {since: 2020}]->(b)<-[:R]-(c)-[]-(d)--(e)<--(f)")
        rels = q.patterns[0].rels
        assert rels[0] == RelPattern("r", ["KNOWS", "LIKES"], {"since": Literal(2020)}, "out")
        assert rels[1] == RelPattern(None, ["R"], {}, "in")
        assert [r.direction for r in rels[2:]] == ["either", "either", "in"]
        assert q.variables == {"a": "node", "r": "relationship", "b": "node", "c": "node",
                               "d": "node", "e": "node", "f": "node"}

    def test_old_type_alternation(self):
        q = parse_query("MATCH ()-[:A|:B]->()")
        assert q.patterns[0].rels[0].types == ["A", "B"]

    def test_null_property_is_kept(self):
        q = parse_query("MATCH (a {k: null, l: [1, -2]})")
        assert q.patterns[0].nodes[0].props == {"k": Literal(None), "l": Literal([1, -2])}

    def test_params(self):
        q = parse_query("MATCH (a {x: $x, y: $`odd name`, z: $0})")
        assert list(q.patterns[0].nodes[0].props.values()) == [
            Param("x"), Param("odd name"), Param("0"),
        ]

    def test_repeated_node_variable_is_a_join(self):
        q = parse_query("MATCH (a)-->(b), (b)-->(a)")
        assert q.variables == {"a": "node", "b": "node"}

    def test_create(self):
        (pattern,) = parse_create("CREATE (a:P {k: null})-[r:R]->(b)")
        assert pattern.nodes[0] == NodePattern("a", [("P",)], {"k": Literal(None)})
        assert pattern.rels[0] == RelPattern("r", ["R"], {}, "out")

    def test_positions(self):
        q = parse_query("MATCH (a)\n  -[r]->(b)")
        assert q.patterns[0].nodes[0].pos == 6
        assert q.patterns[0].rels[0].pos == 12


# =============================================================================
# Expressions
# =============================================================================


class TestExpressions:
    def test_comparisons(self):
        for op in ("=", "<>", "<", "<=", ">", ">=", "=~"):
            assert _where(f"a.x {op} 1") == Compare(op, Prop(Var("a"), "x"), Literal(1))

    def test_arrows_next_to_comparisons(self):
        # `<-` in an expression is `<` then a minus sign
        assert _where("a.x<-1") == Compare("<", Prop(Var("a"), "x"), Literal(-1))
        assert _where("a.x<>-b.y") == Compare("<>", Prop(Var("a"), "x"),
                                              Neg(Prop(Var("b"), "y")))
        assert _where("a.x>=1") == Compare(">=", Prop(Var("a"), "x"), Literal(1))

    def test_chained_comparison(self):
        assert _where("1 < a.x <= 3") == BoolOp("AND", [
            Compare("<", Literal(1), Prop(Var("a"), "x")),
            Compare("<=", Prop(Var("a"), "x"), Literal(3)),
        ])

    def test_precedence(self):
        x, y, z = (Compare("=", Prop(Var("a"), k), Literal(1)) for k in "xyz")
        assert _where("a.x = 1 OR a.y = 1 AND NOT a.z = 1") == BoolOp("OR", [
            x, BoolOp("AND", [y, Not(z)]),
        ])
        assert _where("(a.x = 1 OR a.y = 1) AND a.z = 1") == BoolOp("AND", [
            BoolOp("OR", [x, y]), z,
        ])
        assert _where("a.x = 1 XOR a.y = 1 OR a.z = 1") == BoolOp("OR", [
            BoolOp("XOR", [x, y]), z,
        ])
        assert _where("a.x = 1 AND a.y = 1 AND a.z = 1") == BoolOp("AND", [x, y, z])

    def test_predicates(self):
        name = Prop(Var("a"), "name")
        assert _where("a.name STARTS WITH 'T'") == Compare("STARTS WITH", name, Literal("T"))
        assert _where("a.name ends with $s") == Compare("ENDS WITH", name, Param("s"))
        assert _where("a.name CONTAINS 'o'") == Compare("CONTAINS", name, Literal("o"))
        assert _where("a.name IN ['x', $y]") == Compare(
            "IN", name, ListExpr([Literal("x"), Param("y")]))
        assert _where("a.name IN $names") == Compare("IN", name, Param("names"))
        assert _where("a.name IS NULL") == IsNull(name)
        assert _where("a.name IS NOT NULL") == IsNull(name, negated=True)
        assert _where("NOT a.name IS NULL") == Not(IsNull(name))

    def test_labels_and_functions(self):
        assert _where("a:Person:Actor") == HasLabels(Var("a"), [("Person",), ("Actor",)])
        assert _where("a:A|B") == HasLabels(Var("a"), [("A", "B")])
        assert _where("toLower(a.name) = 'tom'") == Compare(
            "=", Call("toLower", [Prop(Var("a"), "name")]), Literal("tom"))
        assert _where("id(a) <> id(b)") == Compare("<>", Call("id", [Var("a")]),
                                                   Call("id", [Var("b")]))
        assert _where("type(r) = 'R' AND size(labels(a)) > 0").operands[1] == Compare(
            ">", Call("size", [Call("labels", [Var("a")])]), Literal(0))

    def test_literals(self):
        assert _where("a.x = TRUE").right == Literal(True)
        assert _where("a.x = null").right == Literal(None)
        assert _where("a.x = 'it\\'s'").right == Literal("it's")
        assert _where("a.x = 2.5e1").right == Literal(25.0)
        assert _where("a.x = -(1)").right == Literal(-1)
        assert _where("a.x IN []").right == ListExpr([])

    def test_backticked_variable(self):
        q = parse_query("MATCH (`my var`) WHERE `my var`.`a key` = 1 RETURN `my var`")
        assert q.where == Compare("=", Prop(Var("my var"), "a key"), Literal(1))
        assert q.returns.items[0].name == "my var"

    def test_walk(self):
        found = [e for e in walk(_where("a.x = 1 AND NOT toLower(b.y) IN [r.z]"))
                 if isinstance(e, Var)]
        assert [v.name for v in found] == ["a", "b", "r"]


# =============================================================================
# RETURN
# =============================================================================


class TestReturn:
    def test_items(self):
        q = parse_query("MATCH (p)-[r]->(m) RETURN p, r, m.title;")
        assert [item.name for item in q.returns.items] == ["p", "r", "m.title"]
        assert q.returns.items[2].expr == Prop(Var("m"), "title")
        assert not q.returns.distinct

    def test_distinct(self):
        assert parse_query("MATCH (p) RETURN DISTINCT p.name").returns.distinct

    def test_no_where_or_return(self):
        q = parse_query("match (p)")
        assert q.where is None and q.returns is None


# =============================================================================
# Errors
# =============================================================================


class TestErrors:
    @pytest.mark.parametrize(
        "text, message",
        [
            ("", r"starts with MATCH, but the text is empty"),
            ("RETURN 1", r"starts with MATCH, found 'RETURN' \(line 1, column 1\)"),
            ("CREATE (a)", r"CREATE isn't supported: a query only reads the graph"),
            ("OPTIONAL MATCH (a)", r"OPTIONAL MATCH isn't supported"),
            ("MATCH (a) MATCH (b)", r"only one MATCH clause .* \(line 1, column 11\)"),
            ("MATCH (a) RETURN a ORDER BY a.x", r"ORDER isn't supported"),
            ("MATCH (a) RETURN a LIMIT 3", r"LIMIT isn't supported"),
            ("MATCH (a) WITH a", r"WITH isn't supported"),
            ("MATCH (a) b", r"expected WHERE, RETURN or the end of the query, found 'b'"),
            ("MATCH (a) WHERE a.x = 1 (b)",
             r"expected RETURN or the end of the query, found '\('"),
            ("MATCH (a) RETURN a a", r"expected the end of the query, found 'a'"),
            ("MATCH (a)-[*]->(b)", r"variable-length .* select_nodes_in_neighborhood\(\) .*"
                                   r"\(line 1, column 12\)"),
            ("MATCH (a)-[r:R*1..3]->(b)", r"variable-length"),
            ("MATCH (a)-[:A:B]->(b)", r"a single type; write :A\|B"),
            ("MATCH (a)<-[r]->(b)", r"can't point both ways \(line 1, column 10\)"),
            ("MATCH (a)-[r]->(r)", r"`r` is used for both a node and a relationship"
                                   r" \(line 1, column 16\)"),
            ("MATCH (a)-[r]->(b), (b)-[r]->(c)",
             r"relationship variable `r` is used more than once \(line 1, column 24\)"),
            ("MATCH (a) WHERE b.x = 1", r"`b` isn't defined in MATCH \(line 1, column 17\)"),
            ("MATCH (a) RETURN c", r"`c` isn't defined in MATCH \(line 1, column 18\)"),
            ("MATCH (a) WHERE a.x =", r"expected an expression, but the text ended"),
            ("MATCH (a) WHERE AND", r"expected an expression, found 'AND'"),
            ("MATCH (a) WHERE a.x IS 1", r"expected NULL, found '1'"),
            ("MATCH (a) WHERE a.x STARTS 'T'", r"expected WITH, found \"'T'\""),
            ("MATCH (a) WHERE (a.x = 1", r"expected '\)' to close the parenthesis"),
            ("MATCH (a) WHERE a.x IN [1, 2", r"expected ',' or '\]' in the list"),
            ("MATCH (a) WHERE f(a, ", r"expected an expression, but the text ended"),
            ("MATCH (a) WHERE a.x = {k: 1}", r"map values aren't supported"),
            ("MATCH (a) WHERE a.x & 1", r"unexpected character '&'"),
            ("MATCH (a) WHERE a.x = $", r"unexpected character '\$'"),
            ("MATCH (a) RETURN a.x AS y", r"aliases \(AS\) aren't supported"),
            ("MATCH (a) RETURN id(a)", r"only list variables and their properties"),
            ("MATCH (a) RETURN *", r"RETURN \* isn't supported"),
            ("MATCH (a) RETURN a, a", r"the column 'a' is returned twice \(line 1, column 21\)"),
            ("MATCH (a {x: 1, x: 2})", r"'x' is given twice"),
            ("MATCH (a {x: [$y]})", r"expected a value, found '\$y'"),
        ],
    )
    def test_query_errors(self, text, message):
        with pytest.raises(ValueError, match=r"^Cypher: .*" + message):
            parse_query(text)

    @pytest.mark.parametrize(
        "text, message",
        [
            ("CREATE (a {k: $v})", r"parameters \(\$name\) can only be used in queries"
                                   r" \(line 1, column 15\)"),
            ("CREATE (a:A|B)", r"plain labels \(:A:B\), not alternatives \(:A\|B\)"
                               r" \(line 1, column 12\)"),
            ("CREATE (a)-[:A|B]->(b)", r"exactly one type \(line 1, column 15\)"),
            ("CREATE (a)-[*]->(b)", r"expected '\]', found '\*'"),
        ],
    )
    def test_create_errors(self, text, message):
        with pytest.raises(ValueError, match=r"^Cypher: .*" + message):
            parse_create(text)
