"""The Cypher tokenizer, parser and syntax tree.

`from_cypher()` builds graphs from the `CREATE` clauses parsed here, and the query methods run the
`MATCH` queries. Parsing checks the syntax and how variables are used; what a query means (which
functions exist, how values compare) is left to the code that runs it.

This module imports nothing from the rest of EdgeCase, so that both `io` and `graph` can use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

# =============================================================================
# Tokens
# =============================================================================

# `<-` and `->` aren't tokens of their own: a relationship is read from `<`, `-` and `>`, so that
# `a<-1` in an expression is still `a < -1`
_TOKEN = re.compile(
    r"""
      (?P<space>\s+|//[^\n]*|/\*.*?\*/)
    | (?P<name>[^\W\d]\w*)
    | (?P<quoted>`(?:[^`]|``)*`)
    | (?P<string>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
    | (?P<number>(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)
    | (?P<param>\$(?:[^\W\d]\w*|\d+|`(?:[^`]|``)*`))
    | (?P<punct><>|<=|>=|=~|\.\.|[()\[\]{}:,;<>\-.=|*])
    """,
    re.VERBOSE | re.DOTALL,
)

_ESCAPES = {"t": "\t", "b": "\b", "n": "\n", "r": "\r", "f": "\f", "'": "'", '"': '"', "\\": "\\"}

# words that can't be read as a variable inside an expression
_RESERVED = frozenset({
    "AND", "AS", "CONTAINS", "CREATE", "DISTINCT", "ENDS", "IN", "IS", "MATCH", "NOT", "OR",
    "RETURN", "STARTS", "WHERE", "WITH", "XOR",
})

_COMPARISONS = ("=", "<>", "<", "<=", ">", ">=", "=~")


def position_error(fmt: str, text: str, pos: int, message: str) -> ValueError:
    """A `ValueError` for a problem at `pos` in `text`, giving its line and column."""
    line = text.count("\n", 0, pos) + 1
    column = pos - (text.rfind("\n", 0, pos) + 1) + 1
    return ValueError(f"{fmt}: {message} (line {line}, column {column})")


def cypher_error(text: str, pos: int, message: str) -> ValueError:
    """A `ValueError` for a problem at `pos` in the Cypher `text`."""
    return position_error("Cypher", text, pos, message)


# =============================================================================
# Syntax tree
# =============================================================================

# Every node keeps the position of its text, for error messages; positions don't count when
# comparing nodes.


@dataclass
class Literal:
    """A string, number, boolean, `null` or list of those."""

    value: Any
    pos: int = field(default=0, compare=False)


@dataclass
class Param:
    """A query parameter, `$name`."""

    name: str
    pos: int = field(default=0, compare=False)


@dataclass
class Var:
    """A variable bound by the pattern."""

    name: str
    pos: int = field(default=0, compare=False)


@dataclass
class Prop:
    """A property lookup, `subject.key`."""

    subject: Expr
    key: str
    pos: int = field(default=0, compare=False)


@dataclass
class HasLabels:
    """A label test, `subject:A:B` or `subject:A|B` (the same form as `NodePattern.labels`)."""

    subject: Expr
    labels: list[tuple[str, ...]]
    pos: int = field(default=0, compare=False)


@dataclass
class Call:
    """A function call, `name(args)`; the name is kept as written."""

    name: str
    args: list[Expr]
    pos: int = field(default=0, compare=False)


@dataclass
class ListExpr:
    """A list of expressions, `[a, b]`."""

    items: list[Expr]
    pos: int = field(default=0, compare=False)


@dataclass
class Not:
    operand: Expr
    pos: int = field(default=0, compare=False)


@dataclass
class Neg:
    """Unary minus on anything but a number literal (which is folded into the literal)."""

    operand: Expr
    pos: int = field(default=0, compare=False)


@dataclass
class BoolOp:
    """`AND`, `OR` or `XOR` over two or more operands."""

    op: str
    operands: list[Expr]
    pos: int = field(default=0, compare=False)


@dataclass
class Compare:
    """A binary predicate: `=`, `<>`, `<`, `<=`, `>`, `>=`, `=~`, `IN`, `STARTS WITH`,
    `ENDS WITH` or `CONTAINS`. A chain such as `a < b < c` becomes an `AND` of comparisons."""

    op: str
    left: Expr
    right: Expr
    pos: int = field(default=0, compare=False)


@dataclass
class IsNull:
    """`operand IS NULL`, or `IS NOT NULL` when `negated`."""

    operand: Expr
    negated: bool = False
    pos: int = field(default=0, compare=False)


Expr = Union[Literal, Param, Var, Prop, HasLabels, Call, ListExpr, Not, Neg, BoolOp, Compare,
             IsNull]


@dataclass
class NodePattern:
    """`(variable:Label {key: value})`.

    `labels` is a list of alternatives that must all hold: `:A:B` is `[("A",), ("B",)]` and
    `:A|B` is `[("A", "B")]`. Property values are `Literal`s or `Param`s; `null` values are kept.
    """

    variable: Optional[str] = None
    labels: list[tuple[str, ...]] = field(default_factory=list)
    props: dict[str, Expr] = field(default_factory=dict)
    pos: int = field(default=0, compare=False)


@dataclass
class RelPattern:
    """`-[variable:TYPE|OTHER {key: value}]->`, with `direction` `"out"` (`->`), `"in"` (`<-`) or
    `"either"` (no arrow). `types` holds the alternatives (empty for any type)."""

    variable: Optional[str] = None
    types: list[str] = field(default_factory=list)
    props: dict[str, Expr] = field(default_factory=dict)
    direction: str = "out"
    pos: int = field(default=0, compare=False)


@dataclass
class Pattern:
    """A chain of nodes joined by relationships: `rels[k]` joins `nodes[k]` and `nodes[k + 1]`."""

    nodes: list[NodePattern]
    rels: list[RelPattern] = field(default_factory=list)


@dataclass
class ReturnItem:
    """A `RETURN` column: a variable or one of its properties, named as `v` or `v.prop`."""

    expr: Expr
    name: str
    pos: int = field(default=0, compare=False)


@dataclass
class Return:
    items: list[ReturnItem]
    distinct: bool = False


@dataclass
class Query:
    """`MATCH patterns [WHERE where] [RETURN returns]`.

    `variables` maps each named variable to `"node"` or `"relationship"`, in the order they
    first appear.
    """

    patterns: list[Pattern]
    where: Optional[Expr] = None
    returns: Optional[Return] = None
    variables: dict[str, str] = field(default_factory=dict)


def walk(expr: Expr):
    """Yield `expr` and every expression inside it, parents first."""
    yield expr
    if isinstance(expr, (Prop, HasLabels)):
        yield from walk(expr.subject)
    elif isinstance(expr, (Not, Neg, IsNull)):
        yield from walk(expr.operand)
    elif isinstance(expr, Compare):
        yield from walk(expr.left)
        yield from walk(expr.right)
    elif isinstance(expr, (Call, ListExpr, BoolOp)):
        for sub in expr.args if isinstance(expr, Call) else (
            expr.items if isinstance(expr, ListExpr) else expr.operands
        ):
            yield from walk(sub)


# =============================================================================
# Parsing
# =============================================================================


class _TokenParser:
    """The token helpers shared by the text-format parsers (Cypher, DOT)."""

    FORMAT = ""

    def __init__(self, text: str):
        self.text = text
        # (kind, text, position) for every token
        self.tokens: list[tuple[str, Any, int]] = []
        self.i = 0

    def _pos(self) -> int:
        """The position of the next token (the end of the text if there's none)."""
        return self.tokens[self.i][2] if self.i < len(self.tokens) else len(self.text)

    def _fail(self, message: str, pos: Optional[int] = None):
        raise position_error(self.FORMAT, self.text, self._pos() if pos is None else pos, message)

    def _peek(self) -> Optional[str]:
        """The next token's text (`None` at the end)."""
        return self.tokens[self.i][1] if self.i < len(self.tokens) else None

    def _next(self, what: str) -> tuple[str, Any, int]:
        if self.i >= len(self.tokens):
            self._fail(f"expected {what}, but the text ended")
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _expect(self, punct: str, what: Optional[str] = None) -> None:
        tok = self._next(what or repr(punct))
        if tok[1] != punct:
            self.i -= 1
            self._fail(f"expected {what or repr(punct)}, found {tok[1]!r}")


def _unsupported(word: str) -> Optional[str]:
    """Why a clause can't follow the pattern of a query, if `word` starts one."""
    if word in ("CREATE", "MERGE", "SET", "DELETE", "DETACH", "REMOVE"):
        return f"{word} isn't supported: a query only reads the graph"
    if word == "MATCH":
        return "only one MATCH clause is supported; separate the patterns with commas instead"
    if word == "OPTIONAL":
        return "OPTIONAL MATCH isn't supported"
    if word in ("ORDER", "SKIP", "LIMIT"):
        return f"{word} isn't supported; sort or slice the table that match() returns instead"
    if word in ("WITH", "UNWIND", "UNION", "CALL"):
        return f"{word} isn't supported"
    return None


class _CypherParser(_TokenParser):
    """A parser for Cypher `CREATE` scripts and `MATCH` queries."""

    FORMAT = "Cypher"

    def __init__(self, text: str):
        super().__init__(text)
        pos = 0
        while pos < len(text):
            m = _TOKEN.match(text, pos)
            if m is None:
                if text.startswith("/*", pos):
                    self._fail("a comment that never ends", pos)
                if text[pos] in "'\"`":
                    self._fail(f"the quote {text[pos]} is never closed", pos)
                self._fail(f"unexpected character {text[pos]!r}", pos)
            kind = m.lastgroup
            if kind != "space":
                self.tokens.append((kind, m.group(), pos))
            pos = m.end()

    # --- token helpers --------------------------------------------------------

    def _accept(self, punct: str) -> bool:
        if self._peek() == punct and self.tokens[self.i][0] == "punct":
            self.i += 1
            return True
        return False

    def _at_keyword(self, word: str) -> bool:
        return (self.i < len(self.tokens) and self.tokens[self.i][0] == "name"
                and self.tokens[self.i][1].upper() == word)

    def _accept_keyword(self, word: str) -> bool:
        if self._at_keyword(word):
            self.i += 1
            return True
        return False

    def _expect_keyword(self, word: str) -> None:
        if not self._accept_keyword(word):
            if self.i >= len(self.tokens):
                self._fail(f"expected {word}, but the text ended")
            self._fail(f"expected {word}, found {self._peek()!r}")

    def _at_name(self) -> bool:
        return self.i < len(self.tokens) and self.tokens[self.i][0] in ("name", "quoted")

    def _name(self, what: str) -> str:
        kind, value, _ = self._next(what)
        if kind == "name":
            return value
        if kind == "quoted":
            return value[1:-1].replace("``", "`")
        self.i -= 1
        self._fail(f"expected {what}, found {value!r}")

    # --- clauses --------------------------------------------------------------

    def parse_create(self) -> list[Pattern]:
        patterns = []
        while self.i < len(self.tokens):
            if self._accept(";"):
                continue
            kind, word, _ = self._next("CREATE")
            if kind != "name" or word.upper() != "CREATE":
                self.i -= 1
                self._fail(f"only CREATE clauses can build a graph, found {word!r}")
            patterns.append(self._pattern(create=True))
            while self._accept(","):
                patterns.append(self._pattern(create=True))
        return patterns

    def parse_query(self) -> Query:
        if not self._accept_keyword("MATCH"):
            if self.i >= len(self.tokens):
                self._fail("a query starts with MATCH, but the text is empty")
            kind, word, _ = self.tokens[self.i]
            reason = _unsupported(word.upper()) if kind == "name" else None
            if reason is None or word.upper() == "MATCH":
                reason = f"a query starts with MATCH, found {word!r}"
            self._fail(reason)
        patterns = [self._pattern(create=False)]
        while self._accept(","):
            patterns.append(self._pattern(create=False))
        where = self._expr() if self._accept_keyword("WHERE") else None
        returns = self._return() if self._accept_keyword("RETURN") else None
        self._accept(";")
        if self.i < len(self.tokens):
            kind, word, _ = self.tokens[self.i]
            reason = _unsupported(word.upper()) if kind == "name" else None
            if reason is None:
                expected = ("the end of the query" if returns else
                            "RETURN or the end of the query" if where else
                            "WHERE, RETURN or the end of the query")
                reason = f"expected {expected}, found {word!r}"
            self._fail(reason)
        query = Query(patterns, where, returns)
        self._check_variables(query)
        return query

    def _check_variables(self, query: Query) -> None:
        kinds = query.variables
        for pattern in query.patterns:
            parts: list[Union[NodePattern, RelPattern]] = [pattern.nodes[0]]
            for rel, node in zip(pattern.rels, pattern.nodes[1:]):
                parts += [rel, node]
            for part in parts:
                if part.variable is None:
                    continue
                kind = "node" if isinstance(part, NodePattern) else "relationship"
                if part.variable not in kinds:
                    kinds[part.variable] = kind
                elif kinds[part.variable] != kind:
                    self._fail(f"`{part.variable}` is used for both a node and a relationship",
                               part.pos)
                elif kind == "relationship":
                    # a row matches each relationship once, so its variable can't repeat
                    self._fail(f"the relationship variable `{part.variable}` is used more than "
                               "once", part.pos)
        exprs = [query.where] if query.where is not None else []
        if query.returns is not None:
            exprs += [item.expr for item in query.returns.items]
        for expr in exprs:
            for sub in walk(expr):
                if isinstance(sub, Var) and sub.name not in kinds:
                    self._fail(f"the variable `{sub.name}` isn't defined in MATCH", sub.pos)

    def _return(self) -> Return:
        distinct = self._accept_keyword("DISTINCT")
        items: list[ReturnItem] = []
        while True:
            pos = self._pos()
            if self._accept("*"):
                self._fail("RETURN * isn't supported; name the variables to return", pos)
            expr = self._expr()
            if isinstance(expr, Var):
                name = expr.name
            elif isinstance(expr, Prop) and isinstance(expr.subject, Var):
                name = f"{expr.subject.name}.{expr.key}"
            else:
                self._fail("RETURN can only list variables and their properties (v or v.prop)",
                           pos)
            if self._at_keyword("AS"):
                self._fail("aliases (AS) aren't supported; rename the columns of the match() "
                           "table instead")
            if any(item.name == name for item in items):
                self._fail(f"the column {name!r} is returned twice", pos)
            items.append(ReturnItem(expr, name, pos))
            if not self._accept(","):
                return Return(items, distinct)

    # --- patterns -------------------------------------------------------------

    def _pattern(self, create: bool) -> Pattern:
        pattern = Pattern([self._node(create)])
        while self._peek() in ("-", "<"):
            pattern.rels.append(self._rel(create))
            pattern.nodes.append(self._node(create))
        return pattern

    def _node(self, create: bool) -> NodePattern:
        start = self._pos()
        self._expect("(", "'(' to start a node")
        variable = self._name("a variable") if self._at_name() else None
        labels = self._labels(create)
        props = self._map(create) if self._peek() == "{" else {}
        self._expect(")", "')' to end the node")
        return NodePattern(variable, labels, props, start)

    def _labels(self, create: bool) -> list[tuple[str, ...]]:
        labels = []
        while self._accept(":"):
            group = [self._name("a node label")]
            while self._peek() == "|":
                if create:
                    self._fail("a node in CREATE needs plain labels (:A:B), not alternatives "
                               "(:A|B)")
                self.i += 1
                group.append(self._name("a node label"))
            labels.append(tuple(group))
        return labels

    def _rel(self, create: bool) -> RelPattern:
        start = self._pos()
        back = self._accept("<")
        self._expect("-", "'-' in a relationship")
        rel = RelPattern(pos=start)
        if self._accept("["):
            if self._at_name():
                rel.variable = self._name("a relationship variable")
            if self._accept(":"):
                rel.types.append(self._name("a relationship type"))
                while self._peek() == "|":
                    if create:
                        self._fail("a relationship in CREATE needs exactly one type")
                    self.i += 1
                    self._accept(":")  # the older `:A|:B` spelling
                    rel.types.append(self._name("a relationship type"))
                if self._peek() == ":":
                    self._fail("a relationship has a single type; write :A|B to match either")
            if self._peek() == "*" and not create:
                self._fail(
                    "variable-length relationships (*) aren't supported yet; "
                    "select_nodes_in_neighborhood() or trav_out_until() can follow several hops"
                )
            if self._peek() == "{":
                rel.props = self._map(create)
            self._expect("]")
        self._expect("-", "'-' in a relationship")
        forward = self._accept(">")
        if back and forward:
            self._fail("a relationship can't point both ways", start)
        rel.direction = "in" if back else "out" if forward else "either"
        return rel

    def _map(self, create: bool) -> dict[str, Expr]:
        self._expect("{")
        out: dict[str, Expr] = {}
        if self._accept("}"):
            return out
        while True:
            pos = self._pos()
            key = self._name("a property name")
            self._expect(":", "':' after the property name")
            if key in out:
                self._fail(f"the property {key!r} is given twice", pos)
            pos = self._pos()
            if self.i < len(self.tokens) and self.tokens[self.i][0] == "param":
                if create:
                    self._fail("parameters ($name) can only be used in queries")
                out[key] = Param(self._param_name(self.tokens[self.i][1]), pos)
                self.i += 1
            else:
                out[key] = Literal(self._value(), pos)
            if self._accept(","):
                continue
            self._expect("}", "',' or '}' in the properties")
            return out

    def _value(self) -> Any:
        kind, value, pos = self._next("a value")
        if kind == "string":
            return self._unquote(value, pos)
        if kind == "number":
            return _number(value)
        if value == "-" and self._peek() is not None and self.tokens[self.i][0] == "number":
            return -self._value()
        if kind == "name" and value.lower() in ("true", "false", "null"):
            return {"true": True, "false": False, "null": None}[value.lower()]
        if value == "[":
            items = []
            if self._accept("]"):
                return items
            while True:
                items.append(self._value())
                if self._accept(","):
                    continue
                self._expect("]", "',' or ']' in the list")
                return items
        if value == "{":
            self._fail("a property value can't be a map", pos)
        self._fail(f"expected a value, found {value!r}", pos)

    def _unquote(self, token: str, pos: int) -> str:
        body, out, k = token[1:-1], [], 0
        while k < len(body):
            c = body[k]
            if c != "\\":
                out.append(c)
                k += 1
                continue
            e = body[k + 1]
            if e in _ESCAPES:
                out.append(_ESCAPES[e])
                k += 2
            elif e in "uU":
                width = 4 if e == "u" else 8
                digits = body[k + 2:k + 2 + width]
                if not re.fullmatch(r"[0-9a-fA-F]{%d}" % width, digits):
                    self._fail(f"a bad \\{e} escape in a string", pos + 1 + k)
                out.append(chr(int(digits, 16)))
                k += 2 + width
            else:
                self._fail(f"an unknown escape \\{e} in a string", pos + 1 + k)
        return "".join(out)

    @staticmethod
    def _param_name(token: str) -> str:
        name = token[1:]
        return name[1:-1].replace("``", "`") if name.startswith("`") else name

    # --- expressions ----------------------------------------------------------
    #
    # From the loosest binding to the tightest: OR, XOR, AND, NOT, comparisons, the string, list
    # and null predicates, unary minus, then property lookups and label tests.

    def _expr(self) -> Expr:
        return self._bool_op("OR", self._xor)

    def _xor(self) -> Expr:
        return self._bool_op("XOR", self._and)

    def _and(self) -> Expr:
        return self._bool_op("AND", self._not)

    def _bool_op(self, op: str, operand) -> Expr:
        pos = self._pos()
        operands = [operand()]
        while self._accept_keyword(op):
            operands.append(operand())
        return operands[0] if len(operands) == 1 else BoolOp(op, operands, pos)

    def _not(self) -> Expr:
        pos = self._pos()
        if self._accept_keyword("NOT"):
            return Not(self._not(), pos)
        return self._comparison()

    def _comparison(self) -> Expr:
        first = left = self._predicate()
        parts: list[Expr] = []
        while self._peek() in _COMPARISONS and self.tokens[self.i][0] == "punct":
            op, pos = self.tokens[self.i][1], self.tokens[self.i][2]
            self.i += 1
            right = self._predicate()
            parts.append(Compare(op, left, right, pos))
            left = right
        if not parts:
            return first
        return parts[0] if len(parts) == 1 else BoolOp("AND", parts, first.pos)

    def _predicate(self) -> Expr:
        operand = self._unary()
        while True:
            pos = self._pos()
            if self._accept_keyword("IS"):
                negated = self._accept_keyword("NOT")
                self._expect_keyword("NULL")
                operand = IsNull(operand, negated, pos)
            elif self._accept_keyword("IN"):
                operand = Compare("IN", operand, self._unary(), pos)
            elif self._at_keyword("STARTS") or self._at_keyword("ENDS"):
                word = self.tokens[self.i][1].upper()
                self.i += 1
                self._expect_keyword("WITH")
                operand = Compare(f"{word} WITH", operand, self._unary(), pos)
            elif self._accept_keyword("CONTAINS"):
                operand = Compare("CONTAINS", operand, self._unary(), pos)
            else:
                return operand

    def _unary(self) -> Expr:
        pos = self._pos()
        if self._accept("-"):
            operand = self._unary()
            if (isinstance(operand, Literal) and isinstance(operand.value, (int, float))
                    and not isinstance(operand.value, bool)):
                return Literal(-operand.value, pos)
            return Neg(operand, pos)
        return self._postfix()

    def _postfix(self) -> Expr:
        expr = self._atom()
        while True:
            pos = self._pos()
            if self._accept("."):
                expr = Prop(expr, self._name("a property name"), expr.pos)
            elif self._peek() == ":":
                expr = HasLabels(expr, self._labels(create=False), pos)
            else:
                return expr

    def _atom(self) -> Expr:
        kind, value, pos = self._next("an expression")
        if kind == "string":
            return Literal(self._unquote(value, pos), pos)
        if kind == "number":
            return Literal(_number(value), pos)
        if kind == "param":
            return Param(self._param_name(value), pos)
        if kind == "quoted":
            return Var(value[1:-1].replace("``", "`"), pos)
        if kind == "name":
            upper = value.upper()
            if upper in ("TRUE", "FALSE", "NULL"):
                return Literal({"TRUE": True, "FALSE": False, "NULL": None}[upper], pos)
            if self._accept("("):
                args: list[Expr] = []
                if not self._accept(")"):
                    args.append(self._expr())
                    while self._accept(","):
                        args.append(self._expr())
                    self._expect(")", "',' or ')' in the arguments")
                return Call(value, args, pos)
            if upper in _RESERVED:
                self._fail(f"expected an expression, found {value!r}", pos)
            return Var(value, pos)
        if value == "(":
            expr = self._expr()
            self._expect(")", "')' to close the parenthesis")
            return expr
        if value == "[":
            items: list[Expr] = []
            if not self._accept("]"):
                items.append(self._expr())
                while self._accept(","):
                    items.append(self._expr())
                self._expect("]", "',' or ']' in the list")
            return ListExpr(items, pos)
        if value == "{":
            self._fail("map values aren't supported in expressions", pos)
        self._fail(f"expected an expression, found {value!r}", pos)


def _number(text: str) -> Union[int, float]:
    return float(text) if any(c in text for c in ".eE") else int(text)


def parse_create(text: str) -> list[Pattern]:
    """Parse a script of `CREATE` clauses (and `;`s) into its patterns, in order."""
    return _CypherParser(text).parse_create()


def parse_query(text: str) -> Query:
    """Parse a `MATCH ... [WHERE ...] [RETURN ...]` query."""
    return _CypherParser(text).parse_query()
