"""Tests for the Cypher expression evaluator (`edgecase.cypher.Evaluator`) and its value rules."""

import math

import edgecase as ec
import pytest
from edgecase.cypher import (
    Compare,
    Evaluator,
    GraphView,
    HasLabels,
    Literal,
    NodeRef,
    Param,
    Prop,
    RelRef,
    Var,
    conjuncts,
    element_conditions,
    equals,
    labels_of,
    order,
    parse_query,
    variables_in,
)

GRAPH = """
CREATE (keanu:Person:Actor {name: 'Keanu Reeves', born: 1964, nick: null, tags: ['a', 'b']}),
       (matrix:Movie {title: 'The Matrix', released: 1999, rating: 8.7}),
       (keanu)-[:ACTED_IN {roles: ['Neo'], weight: 2}]->(matrix),
       (matrix)-[:LOOP]->(matrix)
"""


@pytest.fixture(scope="module")
def g():
    return ec.from_cypher(GRAPH)


@pytest.fixture(scope="module")
def binding(g):
    acted, _ = sorted(g.get_edge_ids(limit=None))
    return {"a": g.get_id_from_label("Keanu Reeves"), "r": acted,
            "b": g.get_id_from_label("matrix")}


def run(g, binding, expr, params=None, label_key="name"):
    """The value of a RETURN-free expression over `(a)-[r]->(b)`."""
    text = f"MATCH (a)-[r]->(b) WHERE {expr}"
    query = parse_query(text)
    evaluator = Evaluator(text, query.variables, params, GraphView(g._graph, label_key))
    return evaluator.compile(query.clauses[0].where)(binding)


def keeps(g, binding, expr, params=None):
    text = f"MATCH (a)-[r]->(b) WHERE {expr}"
    query = parse_query(text)
    evaluator = Evaluator(text, query.variables, params, GraphView(g._graph))
    return evaluator.condition(query.clauses[0].where)(binding)


# =============================================================================
# Value rules
# =============================================================================


class TestEquals:
    @pytest.mark.parametrize("a, b, expected", [
        (1, 1.0, True), (1, 2, False), ("a", "a", True), ("1", 1, False), (True, 1, False),
        (None, None, None), (None, 1, None), ([1, 2], [1, 2.0], True), ([1], [1, 2], False),
        ([1, None], [1, 2], None), ([1, None], [2, 2], False), ((1, 2), [1, 2], True),
        ({"k": 1}, {"k": 1}, True), ({"k": 1}, {"j": 1}, False), ({"k": None}, {"k": 1}, None),
        (NodeRef(1), NodeRef(1), True), (NodeRef(1), NodeRef(2), False),
        (NodeRef(1), RelRef(1), False), (math.nan, math.nan, False),
    ])
    def test_equals(self, a, b, expected):
        assert equals(a, b) is expected


class TestOrder:
    @pytest.mark.parametrize("a, b, expected", [
        (1, 2.5, -1), (3, 3.0, 0), ("b", "a", 1), (False, True, -1), ([1, 2], [1, 3], -1),
        ([1, 2], [1], 1), ([1, None], [1, 2], None), ([1, None], [2, 2], -1),
        ("3", 4, None), (None, 1, None), (True, 1, None), ({"k": 1}, {"k": 1}, None),
        (NodeRef(1), NodeRef(2), None), (math.nan, 1, None),
    ])
    def test_order(self, a, b, expected):
        c = order(a, b)
        assert (c if c is None else (c > 0) - (c < 0)) == expected


class TestLabelsOf:
    @pytest.mark.parametrize("value, expected", [
        ("Person", ["Person"]), (["A", "B"], ["A", "B"]), (("A",), ["A"]), ([], None),
        (3, None), (["A", 3], None), (None, None),
    ])
    def test_labels_of(self, value, expected):
        assert labels_of(value) == expected


# =============================================================================
# Reading the graph
# =============================================================================


class TestProperties:
    def test_attributes_and_missing(self, g, binding):
        assert run(g, binding, "a.born") == 1964
        assert run(g, binding, "b.title") == "The Matrix"
        assert run(g, binding, "a.tags") == ["a", "b"]
        assert run(g, binding, "a.nope") is None
        assert run(g, binding, "a.nick") is None  # `null` in CREATE means absent

    def test_name_reads_the_label(self, g, binding):
        assert run(g, binding, "a.name") == "Keanu Reeves"
        # the movie has no name, so its label is its variable
        assert run(g, binding, "b.name") == "matrix"
        assert run(g, binding, "a.name", label_key=None) is None
        assert run(g, binding, "a.born", label_key="born") == 1964

    def test_name_attribute_wins_over_the_label(self):
        g = ec.Graph().add_nodes([("x", {"name": "other"})])
        b = {"a": g.get_id_from_label("x"), "r": 0, "b": 0}
        assert run(g, b, "a.name") == "other"

    def test_edge_weight_and_attributes(self, g, binding):
        assert run(g, binding, "r.weight") == 2.0
        assert run(g, binding, "r.roles") == ["Neo"]
        assert run(g, binding, "r.rel") == "ACTED_IN"
        assert run(g, binding, "r.nope") is None

    def test_weight_attribute_when_there_is_no_weight(self):
        g = ec.from_cypher("CREATE (x)-[:R {weight: 'heavy'}]->(y)")
        b = {"a": 0, "r": g.get_edge_ids()[0], "b": 0}
        assert run(g, b, "r.weight") == "heavy"

    def test_labels_and_types(self, g, binding):
        assert run(g, binding, "a:Person") is True
        assert run(g, binding, "a:Person:Actor") is True
        assert run(g, binding, "a:Person:Movie") is False
        assert run(g, binding, "a:Movie|Actor") is True
        assert run(g, binding, "b:Person") is False
        assert run(g, binding, "r:ACTED_IN") is True
        assert run(g, binding, "r:DIRECTED|ACTED_IN") is True
        assert run(g, binding, "r:LOOP") is False

    def test_a_non_label_type_is_no_label(self):
        g = ec.Graph().add_nodes([("x", {"type": 3})])
        b = {"a": g.get_id_from_label("x"), "r": 0, "b": 0}
        assert run(g, b, "labels(a)") == []
        assert run(g, b, "a.type") == 3

    def test_functions(self, g, binding):
        assert run(g, binding, "id(a)") == binding["a"]
        assert run(g, binding, "id(r)") == binding["r"]
        assert run(g, binding, "type(r)") == "ACTED_IN"
        assert run(g, binding, "labels(a)") == ["Person", "Actor"]
        assert run(g, binding, "labels(b)") == ["Movie"]
        assert run(g, binding, "size(a.tags)") == 2
        assert run(g, binding, "size(a.name)") == 12
        assert run(g, binding, "toLower(b.title)") == "the matrix"
        assert run(g, binding, "TOUPPER(b.title)") == "THE MATRIX"
        assert run(g, binding, "exists(a.born)") is True
        assert run(g, binding, "exists(a.nope)") is False
        assert run(g, binding, "size(a.nope)") is None
        assert run(g, binding, "toLower(a.nope)") is None


class TestNodesAsValues:
    def test_identity(self, g, binding):
        assert run(g, binding, "a = a") is True
        assert run(g, binding, "a <> b") is True
        assert run(g, binding, "a = r") is False
        assert run(g, binding, "a IN [b, a]") is True


# =============================================================================
# Operators
# =============================================================================


class TestOperators:
    def test_comparisons(self, g, binding):
        assert run(g, binding, "a.born < 1965") is True
        assert run(g, binding, "a.born >= 1964.0") is True
        assert run(g, binding, "a.born <> 1964") is False
        assert run(g, binding, "1960 < a.born < 1970") is True
        assert run(g, binding, "b.title > 'A'") is True

    def test_mismatched_types_give_null(self, g, binding):
        assert run(g, binding, "a.name < 4") is None
        assert run(g, binding, "'3' < 4") is None
        assert run(g, binding, "a.name = 4") is False

    def test_nan(self, g, binding):
        params = {"nan": math.nan}
        assert run(g, binding, "$nan < 1", params) is False
        assert run(g, binding, "$nan >= 1", params) is False
        assert run(g, binding, "$nan = $nan", params) is False

    def test_strings(self, g, binding):
        assert run(g, binding, "a.name STARTS WITH 'Kea'") is True
        assert run(g, binding, "a.name ENDS WITH 'Reeves'") is True
        assert run(g, binding, "a.name CONTAINS 'nu R'") is True
        assert run(g, binding, "a.name CONTAINS 'x'") is False
        assert run(g, binding, "a.born STARTS WITH '19'") is None
        assert run(g, binding, "a.nope CONTAINS 'x'") is None

    def test_regex(self, g, binding):
        assert run(g, binding, "a.name =~ 'K.*s'") is True
        assert run(g, binding, "a.name =~ 'Kean'") is False  # the whole string must match
        assert run(g, binding, "a.name =~ $re", {"re": "(?i)keanu.*"}) is True
        assert run(g, binding, "a.born =~ '1.*'") is None
        assert run(g, binding, "a.name =~ a.nope") is None

    def test_in(self, g, binding):
        assert run(g, binding, "a.born IN [1964, 1965]") is True
        assert run(g, binding, "a.born IN []") is False
        assert run(g, binding, "a.born IN [1, null]") is None
        assert run(g, binding, "a.born IN [1964, null]") is True
        assert run(g, binding, "a.nope IN [1]") is None
        assert run(g, binding, "a.nope IN []") is False
        assert run(g, binding, "a.born IN a.nope") is None
        assert run(g, binding, "'a' IN a.tags") is True
        assert run(g, binding, "[1, 2] IN [[1, 2]]") is True

    def test_is_null(self, g, binding):
        assert run(g, binding, "a.nope IS NULL") is True
        assert run(g, binding, "a.born IS NULL") is False
        assert run(g, binding, "a.nope IS NOT NULL") is False

    def test_unary_minus(self, g, binding):
        assert run(g, binding, "-a.born") == -1964
        assert run(g, binding, "-a.nope") is None

    @pytest.mark.parametrize("left, right, and_, or_, xor", [
        ("true", "true", True, True, False),
        ("true", "false", False, True, True),
        ("false", "false", False, False, False),
        ("true", "null", None, True, None),
        ("false", "null", False, None, None),
        ("null", "null", None, None, None),
    ])
    def test_three_valued_logic(self, g, binding, left, right, and_, or_, xor):
        for x, y in ((left, right), (right, left)):
            assert run(g, binding, f"{x} AND {y}") is and_
            assert run(g, binding, f"{x} OR {y}") is or_
            assert run(g, binding, f"{x} XOR {y}") is xor

    def test_not(self, g, binding):
        assert run(g, binding, "NOT true") is False
        assert run(g, binding, "NOT null") is None
        assert run(g, binding, "NOT a.nope = 1") is None

    def test_short_circuit(self, g, binding):
        # the right side would be an error, but the left side settles it
        assert run(g, binding, "false AND a.name") is False
        assert run(g, binding, "true OR a.name") is True

    def test_lists(self, g, binding):
        assert run(g, binding, "[a.born, $x]", {"x": "y"}) == [1964, "y"]
        assert run(g, binding, "a.tags = ['a', 'b']") is True
        assert run(g, binding, "a.tags < ['a', 'c']") is True


class TestParams:
    def test_values(self, g, binding):
        assert run(g, binding, "a.born = $year", {"year": 1964}) is True
        assert run(g, binding, "a.name IN $names", {"names": ("Keanu Reeves",)}) is True
        assert run(g, binding, "a.born = $0", {0: 1964}) is True
        assert run(g, binding, "a.born = $0", {"0": 1964}) is True
        assert run(g, binding, "a.born = $`my year`", {"my year": 1964}) is True
        assert run(g, binding, "a.born = $none", {"none": None}) is None

    def test_missing(self, g, binding):
        with pytest.raises(ValueError, match=r"the parameter \$year isn't given in params= "
                                             r"\(line 1, column 35\)"):
            run(g, binding, "a.born = $year")
        with pytest.raises(ValueError, match=r"\$year isn't given"):
            run(g, binding, "a.born = $year", {"other": 1})


class TestCondition:
    def test_true_false_null(self, g, binding):
        assert keeps(g, binding, "a.born = 1964") is True
        assert keeps(g, binding, "a.born = 1") is False
        assert keeps(g, binding, "a.nope = 1") is False

    def test_not_a_boolean(self, g, binding):
        with pytest.raises(ValueError, match=r"WHERE needs true, false or null, not "
                                             r"'Keanu Reeves' \(line 1, column 26\)"):
            keeps(g, binding, "a.name")


# =============================================================================
# Errors
# =============================================================================


class TestErrors:
    @pytest.mark.parametrize("expr, message", [
        ("nope(a)", r"the function nope\(\) isn't supported \(the supported ones are id\(\), "
                    r"type\(\), labels\(\), size\(\), toLower\(\), toUpper\(\), length\(\), nodes\(\), "
                    r"relationships\(\), exists\(\)\) "
                    r"\(line 1, column 26\)"),
        ("tolower(a.x, 1)", r"toLower\(\) takes one argument, not 2"),
        ("id()", r"id\(\) takes one argument, not 0"),
        ("exists(a)", r"exists\(\) takes a property, such as exists\(n.age\) "
                      r"\(line 1, column 33\)"),
        ("a.name =~ '('", r"the regular expression '\(' is invalid"),
    ])
    def test_compile_errors(self, g, binding, expr, message):
        with pytest.raises(ValueError, match=r"^Cypher: " + message):
            run(g, binding, expr)

    @pytest.mark.parametrize("expr, message", [
        ("type(a) = 'x'", r"type\(\) can't take a node \(line 1, column 26\)"),
        ("labels(r)", r"labels\(\) can't take a relationship"),
        ("id(a.born)", r"id\(\) can't take 1964"),
        ("size(a.born)", r"size\(\) can't take 1964"),
        ("toUpper(a.born)", r"toUpper\(\) can't take 1964"),
        ("a.born AND true", r"expected true, false or null, not 1964 \(line 1, column 26\)"),
        ("NOT a.name", r"expected true, false or null, not 'Keanu Reeves'"),
        ("-a.name", r"can't negate 'Keanu Reeves'"),
        ("a.born IN 1964", r"IN needs a list, not 1964"),
        ("a.born.x", r"can't read the property 'x' of 1964"),
        ("a.born:Person", r"only nodes and relationships have labels, not 1964"),
        ("a.name =~ $re", r"the regular expression '\[' is invalid"),
    ])
    def test_runtime_errors(self, g, binding, expr, message):
        with pytest.raises(ValueError, match=r"^Cypher: " + message):
            run(g, binding, expr, {"re": "["})

    def test_map_property(self, g, binding):
        assert run(g, binding, "$m.k", {"m": {"k": 1}}) == 1
        assert run(g, binding, "$m.j", {"m": {"k": 1}}) is None


class TestNullsAndPaths:
    """Variables bound to null (by OPTIONAL MATCH), to lists of relationships (variable-length
    relationships) and to paths (path variables)."""

    def run(self, g, text, binding, params=None):
        query = parse_query(text)
        evaluator = Evaluator(text, query.variables, params, GraphView(g._graph))
        return evaluator.compile(query.clauses[-1].where)(binding)

    @pytest.mark.parametrize("expr, expected", [
        ("a IS NULL", True), ("r IS NULL", True), ("a.name", None), ("r.weight", None),
        ("id(a)", None), ("type(r)", None), ("labels(a)", None), ("a:Person", None),
        ("a = a", None), ("exists(a.name)", False), ("a.name IS NULL", True),
    ])
    def test_null_variables(self, g, expr, expected):
        text = f"MATCH (b) OPTIONAL MATCH (a)-[r]->(b) WHERE {expr}"
        assert self.run(g, text, {"a": None, "r": None, "b": 1}) == expected

    def test_path_values(self, g, binding):
        path = {"a": binding["a"], "r": (binding["r"],), "b": binding["b"]}
        text = "MATCH (a)-[r*]->(b) WHERE {}"
        assert self.run(g, text.format("size(r)"), path) == 1
        assert self.run(g, text.format("size(r)"), {**path, "r": ()}) == 0
        assert self.run(g, text.format("r IS NULL"), {**path, "r": None}) is True
        assert self.run(g, text.format("size(r)"), {**path, "r": None}) is None

    @pytest.mark.parametrize("expr, message", [
        ("id(r) = 1", r"id\(\) can't take a list of relationships"),
        ("type(r) = 'R'", r"type\(\) can't take a list of relationships"),
        ("r:R", r"only nodes and relationships have labels, not a list of relationships"),
    ])
    def test_path_errors(self, g, binding, expr, message):
        path = {"a": binding["a"], "r": (binding["r"],), "b": binding["b"]}
        with pytest.raises(ValueError, match=message):
            self.run(g, f"MATCH (a)-[r*]->(b) WHERE {expr}", path)

    @pytest.fixture
    def walk(self, binding):
        """`p` is the path (a)-[r]->(b)-[loop]->(b)."""
        loop = binding["r"] + 1
        return {**binding, "p": ((binding["a"], binding["b"], binding["b"]), (binding["r"], loop))}

    @pytest.mark.parametrize("expr, expected", [
        ("length(p)", 2), ("size(nodes(p))", 3), ("size(relationships(p))", 2),
        ("nodes(p) = [a, b, b]", True), ("relationships(p) = [r]", False), ("b IN nodes(p)", True),
        ("r IN relationships(p)", True), ("p = p", True), ("p IS NULL", False),
    ])
    def test_path_variables(self, g, walk, expr, expected):
        text = f"MATCH p = (a)-[r]->(b)-[*]->(b) WHERE {expr}"
        assert self.run(g, text, walk) == expected

    def test_null_path_variables(self, g):
        text = "MATCH (b) OPTIONAL MATCH p = (a)-[r]->(b) WHERE {}"
        nulls = {"p": None, "a": None, "r": None, "b": 1}
        for expr in ("length(p)", "nodes(p)", "relationships(p)"):
            assert self.run(g, text.format(expr), nulls) is None
        assert self.run(g, text.format("p IS NULL"), nulls) is True

    @pytest.mark.parametrize("expr, message", [
        ("size(p) = 1", r"size\(\) can't take a path \(line 1, column 39\)"),
        ("id(p) = 1", r"id\(\) can't take a path"),
        ("labels(p) = []", r"labels\(\) can't take a path"),
        ("p:L", r"only nodes and relationships have labels, not a path"),
        ("length(a) = 1", r"length\(\) can't take a node"),
        ("length('abc') = 3", r"length\(\) can't take 'abc'"),
        ("nodes(r) = []", r"nodes\(\) can't take a relationship"),
        ("relationships([1]) = []", r"relationships\(\) can't take \[1\]"),
    ])
    def test_path_variable_errors(self, g, walk, expr, message):
        with pytest.raises(ValueError, match=r"^Cypher: " + message):
            self.run(g, f"MATCH p = (a)-[r]->(b)-[*]->(b) WHERE {expr}", walk)

    def test_path_variable_property_is_an_error_at_once(self, g):
        text = "MATCH p = (a)-->(b) WHERE p.weight > 1"
        query = parse_query(text)
        evaluator = Evaluator(text, query.variables, None, GraphView(g._graph))
        with pytest.raises(ValueError, match=r"`p` is a path, so it has no properties; read them "
                                             r"from the nodes and relationships of its pattern "
                                             r"\(line 1, column 27\)"):
            evaluator.compile(query.clauses[0].where)

    def test_path_property_is_an_error_at_once(self, g):
        text = "MATCH (a)-[r*]->(b) WHERE r.weight > 1"
        query = parse_query(text)
        evaluator = Evaluator(text, query.variables, None, GraphView(g._graph))
        with pytest.raises(ValueError, match=r"`r` is a variable-length relationship, a list of "
                                             r"relationships, so it has no properties.*"
                                             r"\(line 1, column 27\)"):
            evaluator.compile(query.clauses[0].where)


# =============================================================================
# Planner helpers
# =============================================================================


class TestHelpers:
    def test_conjuncts(self):
        q = parse_query("MATCH (a), (b) WHERE a.x = 1 AND (b.y = 2 AND a = b) AND 1 < a.z < 3")
        parts = conjuncts(q.clauses[0].where)
        assert len(parts) == 5
        assert [sorted(variables_in(p)) for p in parts] == [
            ["a"], ["b"], ["a", "b"], ["a"], ["a"],
        ]
        assert conjuncts(None) == []
        assert conjuncts(parse_query("MATCH (a) WHERE a.x OR a.y").clauses[0].where)[0].op == "OR"

    def test_element_conditions(self):
        q = parse_query("MATCH (n:A:B|C {k: 1, p: $p})-[:R|S {w: null}]->()")
        node, rel = q.patterns[0].nodes[0], q.patterns[0].rels[0]
        assert element_conditions("n", node) == [
            HasLabels(Var("n"), [("A",)]), HasLabels(Var("n"), [("B", "C")]),
            Compare("=", Prop(Var("n"), "k"), Literal(1)),
            Compare("=", Prop(Var("n"), "p"), Param("p")),
        ]
        assert element_conditions("r", rel) == [
            HasLabels(Var("r"), [("R", "S")]),
            Compare("=", Prop(Var("r"), "w"), Literal(None)),
        ]
        assert element_conditions("x", q.patterns[0].nodes[1]) == []

    def test_conditions_run(self, g, binding):
        text = "MATCH (a:Person {born: $y})-[r:ACTED_IN|DIRECTED {weight: 2}]->(b {nick: null})"
        q = parse_query(text)
        ev = Evaluator(text, q.variables, {"y": 1964}, GraphView(g._graph))
        pattern = q.patterns[0]
        checks = {
            "a": element_conditions("a", pattern.nodes[0]),
            "r": element_conditions("r", pattern.rels[0]),
            "b": element_conditions("b", pattern.nodes[1]),
        }
        results = {v: [ev.condition(c)(binding) for c in cs] for v, cs in checks.items()}
        # `{nick: null}` never matches, as in Neo4j
        assert results == {"a": [True, True], "r": [True, True], "b": [False]}
