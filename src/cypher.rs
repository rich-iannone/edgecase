//! The Cypher matcher: the backtracking search that `Matcher` in `cypher.py` runs over a plan.
//!
//! Python parses and plans the query and compiles every condition into a Python function. It
//! hands the plan over as a *program* (see `Matcher._compile()`), where each condition may also
//! carry a *native* form: a label or type test, a comparison of a property or an ID with a
//! constant, `IN` a constant list, `IS NULL`, or a comparison of two IDs. The search checks a
//! native form here, and calls the Python function only for the other conditions and for values
//! the native form can't decide (an attribute value of an unusual type, say). Either way the
//! answer is the one the Python evaluator gives, and the matches come in the same order as from
//! the Python search, which stays as the reference.

use crate::graph::Graph;
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use rustc_hash::FxHashSet;
use std::rc::Rc;

/// The value of a slot during the search
#[derive(Clone)]
enum Val {
    Unset,
    Null,
    Node(u64),
    Edge(u64),
    Path(Vec<u64>),
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Node,
    Rel,
    Path,
}

#[derive(Clone, Copy, PartialEq)]
enum Dir {
    Out,
    In,
    Both,
}

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Op {
    fn parse(text: &str) -> PyResult<Op> {
        Ok(match text {
            "=" => Op::Eq,
            "<>" => Op::Ne,
            "<" => Op::Lt,
            "<=" => Op::Le,
            ">" => Op::Gt,
            ">=" => Op::Ge,
            _ => return Err(bad_program(&format!("unknown operator {text:?}"))),
        })
    }

    fn compare_op(self) -> CompareOp {
        match self {
            Op::Eq => CompareOp::Eq,
            Op::Ne => CompareOp::Ne,
            Op::Lt => CompareOp::Lt,
            Op::Le => CompareOp::Le,
            Op::Gt => CompareOp::Gt,
            Op::Ge => CompareOp::Ge,
        }
    }

    fn holds<T: PartialOrd>(self, a: T, b: T) -> bool {
        match self {
            Op::Eq => a == b,
            Op::Ne => a != b,
            Op::Lt => a < b,
            Op::Le => a <= b,
            Op::Gt => a > b,
            Op::Ge => a >= b,
        }
    }
}

/// The kind of a value, as `_kind()` in `cypher.py` sees it. `Other` is anything the native
/// forms leave to Python.
#[derive(Clone, Copy, PartialEq)]
enum Class {
    Null,
    Bool,
    Number,
    Str,
    List,
    Map,
    Other,
}

fn classify(value: &Bound<'_, PyAny>) -> Class {
    if value.is_none() {
        Class::Null
    } else if value.is_instance_of::<PyBool>() {
        Class::Bool
    } else if value.is_instance_of::<PyInt>() || value.is_instance_of::<PyFloat>() {
        Class::Number
    } else if value.is_instance_of::<PyString>() {
        Class::Str
    } else if value.is_instance_of::<PyList>() || value.is_instance_of::<PyTuple>() {
        Class::List
    } else if value.is_instance_of::<PyDict>() {
        Class::Map
    } else {
        Class::Other
    }
}

/// What a comparison reads from the binding
enum Operand {
    /// A property of a node or relationship slot, with `key == label_key` (for nodes) or
    /// `key == "weight"` (for relationships) noted
    Prop {
        slot: usize,
        key: Py<PyString>,
        special: bool,
    },
    /// The ID of a node or relationship slot
    Id(usize),
}

/// A condition the search can check without Python
enum Native {
    /// `x:A:B|C` for a node, or `r:A|B` for a relationship: every group must hold one label
    Labels(usize, Vec<Vec<Py<PyString>>>),
    /// `operand op constant`, for a constant boolean, number or string
    Compare(Operand, Op, Py<PyAny>, Class),
    /// `operand IN [constants]`, each a boolean, number, string or `null`
    In(Operand, Vec<(Py<PyAny>, Class)>),
    /// `slot IS NULL` (no key) or `slot.key IS NULL`, or with `negated`, `IS NOT NULL`
    IsNull(usize, Option<(Py<PyString>, bool)>, bool),
    /// `id(a) op id(b)`, also for `a = b` and `a <> b`
    Ids(usize, usize, Op),
    /// A condition that is never true, such as `{k: null}`
    Never,
}

struct Cond {
    native: Option<Native>,
    fallback: Py<PyAny>,
}

#[derive(PartialEq)]
enum StepKind {
    Scan,
    Expand,
    Connect,
}

struct Step {
    kind: StepKind,
    node: usize,
    rel: usize,
    start: usize,
    dir: Dir,
    seed: Option<Py<PyAny>>,
    conds: Vec<Cond>,
    hops: Option<(usize, Option<usize>)>,
    edge_conds: Vec<Cond>,
    reverse: bool,
    fixed: bool,
}

struct Stage {
    optional: bool,
    entry: Vec<Cond>,
    new: Vec<usize>,
    joined: Vec<usize>,
    steps: Vec<Step>,
}

struct Program {
    names: Vec<Py<PyString>>,
    kinds: Vec<Kind>,
    /// The slot the conditions on each edge of a variable-length relationship read
    edge_slot: usize,
    distinct_nodes: bool,
    stages: Vec<Stage>,
}

fn bad_program(message: &str) -> PyErr {
    PyValueError::new_err(format!("invalid Cypher match program: {message}"))
}

// =============================================================================
// Reading the program
// =============================================================================

fn parse_program(program: &Bound<'_, PyAny>) -> PyResult<Program> {
    type Fields<'py> = (
        Vec<String>,
        Vec<String>,
        Option<String>,
        bool,
        Vec<Bound<'py, PyAny>>,
    );
    let (names, kinds, label_key, distinct_nodes, stages): Fields = program.extract()?;
    let py = program.py();
    let kinds = kinds
        .iter()
        .map(|kind| match kind.as_str() {
            "node" => Ok(Kind::Node),
            "relationship" => Ok(Kind::Rel),
            "relationships" => Ok(Kind::Path),
            _ => Err(bad_program(&format!("unknown slot kind {kind:?}"))),
        })
        .collect::<PyResult<Vec<Kind>>>()?;
    if names.is_empty() || names.len() != kinds.len() {
        return Err(bad_program("the slot names and kinds don't match"));
    }
    let reader = Reader {
        kinds: &kinds,
        label_key: label_key.as_deref(),
    };
    let stages = stages
        .iter()
        .map(|stage| reader.stage(stage))
        .collect::<PyResult<_>>()?;
    Ok(Program {
        edge_slot: names.len() - 1,
        names: names
            .iter()
            .map(|name| PyString::new(py, name).unbind())
            .collect(),
        kinds,
        distinct_nodes,
        stages,
    })
}

struct Reader<'k> {
    kinds: &'k [Kind],
    label_key: Option<&'k str>,
}

impl Reader<'_> {
    fn slot(&self, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        let slot: usize = item.extract()?;
        if slot >= self.kinds.len() {
            return Err(bad_program(&format!("slot {slot} is out of range")));
        }
        Ok(slot)
    }

    fn stage(&self, stage: &Bound<'_, PyAny>) -> PyResult<Stage> {
        type Items<'py> = Vec<Bound<'py, PyAny>>;
        type Fields<'py> = (bool, Items<'py>, Items<'py>, Items<'py>, Items<'py>);
        let (optional, entry, new, joined, steps): Fields = stage.extract()?;
        Ok(Stage {
            optional,
            entry: self.conds(&entry)?,
            new: new.iter().map(|s| self.slot(s)).collect::<PyResult<_>>()?,
            joined: joined
                .iter()
                .map(|s| self.slot(s))
                .collect::<PyResult<_>>()?,
            steps: steps
                .iter()
                .map(|step| self.step(step))
                .collect::<PyResult<_>>()?,
        })
    }

    fn step(&self, step: &Bound<'_, PyAny>) -> PyResult<Step> {
        type Fields<'py> = (
            String,
            Bound<'py, PyAny>,
            Option<Bound<'py, PyAny>>,
            Option<Bound<'py, PyAny>>,
            Option<String>,
            Option<Py<PyAny>>,
            Vec<Bound<'py, PyAny>>,
            Option<(usize, Option<usize>)>,
            Vec<Bound<'py, PyAny>>,
            bool,
            bool,
        );
        let (kind, node, rel, start, dir, seed, conds, hops, edge_conds, reverse, fixed): Fields =
            step.extract()?;
        let kind = match kind.as_str() {
            "scan" => StepKind::Scan,
            "expand" => StepKind::Expand,
            "connect" => StepKind::Connect,
            _ => return Err(bad_program(&format!("unknown step kind {kind:?}"))),
        };
        let dir = match dir.as_deref() {
            None | Some("both") => Dir::Both,
            Some("out") => Dir::Out,
            Some("in") => Dir::In,
            Some(other) => return Err(bad_program(&format!("unknown direction {other:?}"))),
        };
        let optional_slot = |item: Option<Bound<'_, PyAny>>| -> PyResult<usize> {
            match item {
                Some(item) => self.slot(&item),
                None if kind == StepKind::Scan => Ok(0),
                None => Err(bad_program("an expand or connect step needs rel and start")),
            }
        };
        let (rel, start) = (optional_slot(rel)?, optional_slot(start)?);
        Ok(Step {
            node: self.slot(&node)?,
            rel,
            start,
            dir,
            seed,
            conds: self.conds(&conds)?,
            hops,
            edge_conds: self.conds(&edge_conds)?,
            reverse,
            fixed,
            kind,
        })
    }

    fn conds(&self, conds: &[Bound<'_, PyAny>]) -> PyResult<Vec<Cond>> {
        conds
            .iter()
            .map(|cond| {
                let (native, fallback): (Option<Bound<'_, PyAny>>, Py<PyAny>) = cond.extract()?;
                let native = native.map(|n| self.native(&n)).transpose()?;
                Ok(Cond { native, fallback })
            })
            .collect()
    }

    fn native(&self, native: &Bound<'_, PyAny>) -> PyResult<Native> {
        let tag: String = native.get_item(0)?.extract()?;
        let item = |i: usize| native.get_item(i);
        Ok(match tag.as_str() {
            "labels" => {
                let slot = self.single(&item(1)?)?;
                let groups: Vec<Vec<Bound<'_, PyString>>> = item(2)?.extract()?;
                let groups = groups
                    .into_iter()
                    .map(|group| group.into_iter().map(Bound::unbind).collect())
                    .collect();
                Native::Labels(slot, groups)
            }
            "cmp" => {
                let value = item(3)?;
                let class = self.constant(&value)?;
                if class == Class::Null {
                    Native::Never
                } else {
                    let op = Op::parse(&item(2)?.extract::<String>()?)?;
                    Native::Compare(self.operand(&item(1)?)?, op, value.unbind(), class)
                }
            }
            "in" => {
                let values: Vec<Bound<'_, PyAny>> = item(2)?.extract()?;
                let values = values
                    .into_iter()
                    .map(|v| Ok((self.constant(&v)?, v)))
                    .map(|found: PyResult<_>| found.map(|(class, v)| (v.unbind(), class)))
                    .collect::<PyResult<_>>()?;
                Native::In(self.operand(&item(1)?)?, values)
            }
            "null" => {
                let slot = self.slot(&item(1)?)?;
                let key: Option<Bound<'_, PyString>> = item(2)?.extract()?;
                let key = match key {
                    Some(key) => {
                        let (key, special) = self.prop_key(slot, key)?;
                        Some((key, special))
                    }
                    None => None,
                };
                Native::IsNull(slot, key, item(3)?.extract()?)
            }
            "ids" => {
                let op = Op::parse(&item(3)?.extract::<String>()?)?;
                Native::Ids(self.single(&item(1)?)?, self.single(&item(2)?)?, op)
            }
            "never" => Native::Never,
            _ => return Err(bad_program(&format!("unknown native condition {tag:?}"))),
        })
    }

    /// A node or (single) relationship slot
    fn single(&self, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        let slot = self.slot(item)?;
        if self.kinds[slot] == Kind::Path {
            return Err(bad_program(
                "a variable-length relationship can't be tested natively",
            ));
        }
        Ok(slot)
    }

    fn operand(&self, operand: &Bound<'_, PyAny>) -> PyResult<Operand> {
        let tag: String = operand.get_item(0)?.extract()?;
        let slot = self.single(&operand.get_item(1)?)?;
        match tag.as_str() {
            "prop" => {
                let (key, special) = self.prop_key(slot, operand.get_item(2)?.extract()?)?;
                Ok(Operand::Prop { slot, key, special })
            }
            "id" => Ok(Operand::Id(slot)),
            _ => Err(bad_program(&format!("unknown operand {tag:?}"))),
        }
    }

    fn prop_key(&self, slot: usize, key: Bound<'_, PyString>) -> PyResult<(Py<PyString>, bool)> {
        if self.kinds[slot] == Kind::Path {
            return Err(bad_program(
                "a variable-length relationship has no properties",
            ));
        }
        let text = key.to_cow()?;
        let special = match self.kinds[slot] {
            Kind::Node => self.label_key == Some(text.as_ref()),
            _ => text == "weight",
        };
        Ok((key.unbind(), special))
    }

    /// The class of a constant, which must be `null`, a boolean, a number or a string
    fn constant(&self, value: &Bound<'_, PyAny>) -> PyResult<Class> {
        match classify(value) {
            class @ (Class::Null | Class::Bool | Class::Number | Class::Str) => Ok(class),
            _ => Err(bad_program(
                "a native constant must be null, a boolean, a number or a string",
            )),
        }
    }
}

// =============================================================================
// Searching
// =============================================================================

/// One level of the depth-first walk over the paths of a variable-length relationship
struct Frame {
    node: u64,
    /// The position of the next incident edge to try
    next: usize,
    /// Whether stepping onto `node` added it to the used nodes
    marked: bool,
}

struct Search<'a, 'py> {
    py: Python<'py>,
    graph: &'a Graph,
    program: &'a Program,
    binding: Vec<Val>,
    /// The nodes and edges used so far by each stage's clause
    used_nodes: Vec<FxHashSet<u64>>,
    used_edges: Vec<FxHashSet<u64>>,
    /// Whether each stage has found a match for the current row of the stages before it
    found: Vec<bool>,
    all_nodes: Option<Rc<Vec<u64>>>,
    /// The candidates of each seeded scan, by stage and step, worked out on first use
    seeded: Vec<Vec<Option<Rc<Vec<u64>>>>>,
    limit: Option<usize>,
    count_only: bool,
    count: usize,
    rows: Vec<Py<PyTuple>>,
    first: Option<Py<PyTuple>>,
    ticks: u32,
}

/// The incident edges of `node` in `dir`, from position `*next` on, as `_incident_edges()`
/// lists them: for `Both`, the out-edges and then the in-edges, each self-loop once
fn next_pair(graph: &Graph, node: u64, dir: Dir, next: &mut usize) -> Option<(u64, u64)> {
    let out = if dir == Dir::In {
        &[][..]
    } else {
        graph.out_pairs(node)
    };
    let inward = if dir == Dir::Out {
        &[][..]
    } else {
        graph.in_pairs(node)
    };
    loop {
        let i = *next;
        *next += 1;
        if i < out.len() {
            return Some(out[i]);
        }
        let pair = *inward.get(i - out.len())?;
        if dir == Dir::Both && pair.1 == node {
            continue; // a self-loop, already listed as an out-edge
        }
        return Some(pair);
    }
}

impl<'a, 'py> Search<'a, 'py> {
    fn node_at(&self, slot: usize) -> PyResult<u64> {
        match self.binding[slot] {
            Val::Node(node) => Ok(node),
            _ => Err(bad_program(
                "a step starts or ends at a slot that holds no node",
            )),
        }
    }

    fn value(&self, val: &Val) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        Ok(match val {
            Val::Unset | Val::Null => py.None().into_bound(py),
            Val::Node(id) | Val::Edge(id) => (*id).into_pyobject(py)?.into_any(),
            Val::Path(edges) => PyTuple::new(py, edges.iter().copied())?.into_any(),
        })
    }

    /// The current match, as a tuple in slot order (without the edge slot)
    fn row(&self) -> PyResult<Py<PyTuple>> {
        let values = self.binding[..self.program.edge_slot]
            .iter()
            .map(|val| self.value(val))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(self.py, values)?.unbind())
    }

    /// Record a match; `true` means the limit is reached
    fn emit(&mut self) -> PyResult<bool> {
        self.count += 1;
        if !self.count_only {
            let row = self.row()?;
            self.rows.push(row);
        } else if self.first.is_none() {
            self.first = Some(self.row()?);
        }
        Ok(self.limit.is_some_and(|limit| self.count >= limit))
    }

    /// Let Ctrl-C stop a long search
    fn tick(&mut self) -> PyResult<()> {
        self.ticks = self.ticks.wrapping_add(1);
        if self.ticks % 65536 == 0 {
            self.py.check_signals()?;
        }
        Ok(())
    }

    // --- conditions -----------------------------------------------------------

    fn check_all(&self, conds: &[Cond]) -> PyResult<bool> {
        for cond in conds {
            if !self.check(cond)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn check(&self, cond: &Cond) -> PyResult<bool> {
        if let Some(native) = &cond.native {
            if let Some(result) = self.native(native) {
                return Ok(result);
            }
        }
        let binding = PyDict::new(self.py);
        for (val, name) in self.binding.iter().zip(&self.program.names) {
            if !matches!(val, Val::Unset) {
                binding.set_item(name.bind(self.py), self.value(val)?)?;
            }
        }
        cond.fallback.bind(self.py).call1((binding,))?.is_truthy()
    }

    /// The result of a native condition, or `None` when it's for Python to decide (Python
    /// errors are also left to Python, which raises them the same way)
    fn native(&self, native: &Native) -> Option<bool> {
        let py = self.py;
        match native {
            Native::Labels(slot, groups) => self.has_labels(*slot, groups).ok(),
            Native::Compare(operand, op, constant, class) => {
                let value = self.operand(operand).ok()?;
                compare(&value, *op, constant.bind(py), *class)
            }
            Native::In(operand, constants) => {
                let value = self.operand(operand).ok()?;
                match classify(&value) {
                    Class::Other => None,
                    Class::Null => Some(false),
                    _ => {
                        for (constant, class) in constants {
                            if *class != Class::Null
                                && compare(&value, Op::Eq, constant.bind(py), *class)?
                            {
                                return Some(true);
                            }
                        }
                        Some(false)
                    }
                }
            }
            Native::IsNull(slot, key, negated) => {
                let null = match (&self.binding[*slot], key) {
                    (Val::Null | Val::Unset, _) => true,
                    (_, None) => false,
                    (_, Some((key, special))) => self.prop(*slot, key, *special).ok()?.is_none(),
                };
                Some(null != *negated)
            }
            Native::Ids(a, b, op) => match (&self.binding[*a], &self.binding[*b]) {
                (Val::Node(x) | Val::Edge(x), Val::Node(y) | Val::Edge(y)) => Some(op.holds(x, y)),
                _ => Some(false),
            },
            Native::Never => Some(false),
        }
    }

    fn operand(&self, operand: &Operand) -> PyResult<Bound<'py, PyAny>> {
        match operand {
            Operand::Prop { slot, key, special } => self.prop(*slot, key, *special),
            Operand::Id(slot) => match self.binding[*slot] {
                Val::Node(id) | Val::Edge(id) => Ok(id.into_pyobject(self.py)?.into_any()),
                _ => Ok(self.py.None().into_bound(self.py)),
            },
        }
    }

    /// A property, as `GraphView.node_prop()` and `edge_prop()` read it
    fn prop(&self, slot: usize, key: &Py<PyString>, special: bool) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let none = || py.None().into_bound(py);
        match self.binding[slot] {
            Val::Node(id) => {
                let node = self
                    .graph
                    .node_by_id(id)
                    .ok_or_else(|| bad_program("no such node"))?;
                if let Some(attrs) = &node.attributes {
                    if let Some(value) = attrs.bind(py).get_item(key.bind(py))? {
                        return Ok(value);
                    }
                }
                Ok(match (&node.label, special) {
                    (Some(label), true) => PyString::new(py, label).into_any(),
                    _ => none(),
                })
            }
            Val::Edge(id) => {
                let edge = self
                    .graph
                    .edge_by_id(id)
                    .ok_or_else(|| bad_program("no such edge"))?;
                if let (Some(weight), true) = (edge.weight, special) {
                    return Ok(PyFloat::new(py, weight).into_any());
                }
                if let Some(attrs) = &edge.attributes {
                    if let Some(value) = attrs.bind(py).get_item(key.bind(py))? {
                        return Ok(value);
                    }
                }
                Ok(none())
            }
            _ => Ok(none()),
        }
    }

    fn has_labels(&self, slot: usize, groups: &[Vec<Py<PyString>>]) -> PyResult<bool> {
        let py = self.py;
        let have: Vec<Bound<'py, PyAny>> = match self.binding[slot] {
            Val::Node(id) => {
                let node = self
                    .graph
                    .node_by_id(id)
                    .ok_or_else(|| bad_program("no such node"))?;
                let value = match &node.attributes {
                    Some(attrs) => attrs.bind(py).get_item("type")?,
                    None => None,
                };
                // as `labels_of()`: a string, or a non-empty list or tuple of strings
                match value {
                    Some(v) if v.is_instance_of::<PyString>() => vec![v],
                    Some(v) if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() => {
                        let items: Vec<Bound<'py, PyAny>> =
                            v.try_iter()?.collect::<PyResult<_>>()?;
                        if items.iter().all(|item| item.is_instance_of::<PyString>()) {
                            items
                        } else {
                            vec![]
                        }
                    }
                    _ => vec![],
                }
            }
            Val::Edge(id) => {
                let edge = self
                    .graph
                    .edge_by_id(id)
                    .ok_or_else(|| bad_program("no such edge"))?;
                let value = match &edge.attributes {
                    Some(attrs) => attrs.bind(py).get_item("rel")?,
                    None => None,
                };
                value
                    .filter(|v| v.is_instance_of::<PyString>())
                    .into_iter()
                    .collect()
            }
            _ => return Ok(false), // null
        };
        for group in groups {
            let mut any = false;
            for label in group {
                for item in &have {
                    if item.eq(label.bind(py))? {
                        any = true;
                        break;
                    }
                }
                if any {
                    break;
                }
            }
            if !any {
                return Ok(false);
            }
        }
        Ok(true)
    }

    // --- the search -------------------------------------------------------------

    fn stage(&mut self, s: usize) -> PyResult<bool> {
        let program = self.program;
        if s == program.stages.len() {
            return self.emit();
        }
        let stage = &program.stages[s];
        self.found[s] = false;
        if self.enter(s)? && self.search(s, 0)? {
            return Ok(true);
        }
        if stage.optional && !self.found[s] {
            for &slot in &stage.new {
                self.binding[slot] = Val::Null;
            }
            return self.stage(s + 1);
        }
        Ok(false)
    }

    /// Check a stage's entry conditions and the slots it shares with earlier stages, and start
    /// its used nodes and edges with those slots
    fn enter(&mut self, s: usize) -> PyResult<bool> {
        let program = self.program;
        let stage = &program.stages[s];
        if !self.check_all(&stage.entry)? {
            return Ok(false);
        }
        let distinct = program.distinct_nodes;
        self.used_nodes[s].clear();
        self.used_edges[s].clear();
        for &slot in &stage.joined {
            match self.binding[slot] {
                Val::Node(node) => {
                    if !self.used_nodes[s].insert(node) && distinct {
                        return Ok(false);
                    }
                }
                Val::Edge(edge) => {
                    if !self.used_edges[s].insert(edge) {
                        return Ok(false);
                    }
                }
                _ => return Ok(false), // null: a pattern that uses it again matches nothing
            }
        }
        if !distinct {
            self.used_nodes[s].clear();
        }
        // the seeds are worked out on entering the stage, as in the Python search, so that an
        // error in one raises at the same point
        for k in 0..stage.steps.len() {
            if stage.steps[k].seed.is_some() {
                self.candidates(s, k)?;
            }
        }
        Ok(true)
    }

    fn candidates(&mut self, s: usize, k: usize) -> PyResult<Rc<Vec<u64>>> {
        let program = self.program;
        if let Some(seed) = &program.stages[s].steps[k].seed {
            if let Some(found) = &self.seeded[s][k] {
                return Ok(found.clone());
            }
            let ids: Vec<u64> = seed.bind(self.py).call0()?.extract()?;
            let ids = Rc::new(ids);
            self.seeded[s][k] = Some(ids.clone());
            return Ok(ids);
        }
        Ok(self
            .all_nodes
            .get_or_insert_with(|| Rc::new(self.graph.sorted_node_ids()))
            .clone())
    }

    /// Bind the slots of step `k` of stage `s` and those after it in every way that matches;
    /// `true` means the limit is reached
    fn search(&mut self, s: usize, k: usize) -> PyResult<bool> {
        let program = self.program;
        let steps = &program.stages[s].steps;
        if k == steps.len() {
            self.found[s] = true;
            return self.stage(s + 1);
        }
        self.tick()?;
        let step = &steps[k];
        let distinct = program.distinct_nodes;
        if step.kind == StepKind::Scan {
            let candidates = self.candidates(s, k)?;
            for &node in candidates.iter() {
                if distinct && self.used_nodes[s].contains(&node) {
                    continue;
                }
                self.binding[step.node] = Val::Node(node);
                if self.check_all(&step.conds)? {
                    if distinct {
                        self.used_nodes[s].insert(node);
                    }
                    if self.search(s, k + 1)? {
                        return Ok(true);
                    }
                    if distinct {
                        self.used_nodes[s].remove(&node);
                    }
                }
            }
            return Ok(false);
        }
        if step.hops.is_some() {
            return self.paths(s, k);
        }
        let graph = self.graph;
        let connect = step.kind == StepKind::Connect;
        let start = self.node_at(step.start)?;
        let target = if connect {
            Some(self.node_at(step.node)?)
        } else {
            None
        };
        let fixed = match (step.fixed, &self.binding[step.rel]) {
            (false, _) => None,
            (true, Val::Edge(edge)) => Some(*edge),
            (true, _) => return Ok(false),
        };
        let mut next = 0;
        while let Some((edge, other)) = next_pair(graph, start, step.dir, &mut next) {
            match fixed {
                Some(fixed) if edge != fixed => continue,
                None if self.used_edges[s].contains(&edge) => continue,
                _ => {}
            }
            match target {
                Some(target) if other != target => continue,
                Some(_) => {}
                None if distinct && self.used_nodes[s].contains(&other) => continue,
                None => self.binding[step.node] = Val::Node(other),
            }
            self.binding[step.rel] = Val::Edge(edge);
            if !self.check_all(&step.conds)? {
                continue;
            }
            let fresh = distinct && !connect;
            if fixed.is_none() {
                self.used_edges[s].insert(edge);
            }
            if fresh {
                self.used_nodes[s].insert(other);
            }
            if self.search(s, k + 1)? {
                return Ok(true);
            }
            if fixed.is_none() {
                self.used_edges[s].remove(&edge);
            }
            if fresh {
                self.used_nodes[s].remove(&other);
            }
        }
        Ok(false)
    }

    /// The paths of a variable-length step: a depth-first walk from its start, using each edge
    /// once per clause, kept on an explicit stack so long paths can't overflow the call stack
    fn paths(&mut self, s: usize, k: usize) -> PyResult<bool> {
        let program = self.program;
        let step = &program.stages[s].steps[k];
        let (low, high) = step.hops.unwrap_or((1, Some(1)));
        let distinct = program.distinct_nodes;
        let connect = step.kind == StepKind::Connect;
        let target = if connect {
            Some(self.node_at(step.node)?)
        } else {
            None
        };
        let start = self.node_at(step.start)?;
        let mut path: Vec<u64> = Vec::new();
        if self.path_end(s, k, start, &path, low)? {
            return Ok(true);
        }
        let mut stack = vec![Frame {
            node: start,
            next: 0,
            marked: false,
        }];
        while !stack.is_empty() {
            let depth = stack.len() - 1;
            let top = &mut stack[depth];
            let node = top.node;
            // with distinct nodes, a path can end at the target, but not pass through it
            let cut = high == Some(depth) || (distinct && depth > 0 && target == Some(node));
            let pair = if cut {
                None
            } else {
                next_pair(self.graph, node, step.dir, &mut top.next)
            };
            let Some((edge, other)) = pair else {
                let frame = stack.pop().expect("the stack isn't empty");
                if !stack.is_empty() {
                    if let Some(edge) = path.pop() {
                        self.used_edges[s].remove(&edge);
                    }
                    if frame.marked {
                        self.used_nodes[s].remove(&frame.node);
                    }
                }
                continue;
            };
            if self.used_edges[s].contains(&edge) {
                continue;
            }
            let mark = distinct && !self.used_nodes[s].contains(&other);
            if distinct && !mark && target != Some(other) {
                continue;
            }
            self.binding[program.edge_slot] = Val::Edge(edge);
            if !self.check_all(&step.edge_conds)? {
                continue;
            }
            self.used_edges[s].insert(edge);
            path.push(edge);
            if mark {
                self.used_nodes[s].insert(other);
            }
            stack.push(Frame {
                node: other,
                next: 0,
                marked: mark,
            });
            self.tick()?;
            if self.path_end(s, k, other, &path, low)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Try the walk's current path, which ends at `node`, as a match of the step
    fn path_end(
        &mut self,
        s: usize,
        k: usize,
        node: u64,
        path: &[u64],
        low: usize,
    ) -> PyResult<bool> {
        let program = self.program;
        let step = &program.stages[s].steps[k];
        let connect = step.kind == StepKind::Connect;
        if path.len() < low {
            return Ok(false);
        }
        if connect {
            if self.node_at(step.node)? != node {
                return Ok(false);
            }
        } else if program.distinct_nodes && path.is_empty() {
            return Ok(false); // the end would be the start
        } else {
            self.binding[step.node] = Val::Node(node);
        }
        let mut edges = path.to_vec();
        if step.reverse {
            edges.reverse();
        }
        self.binding[step.rel] = Val::Path(edges);
        if self.check_all(&step.conds)? {
            return self.search(s, k + 1);
        }
        Ok(false)
    }
}

/// `value op constant` by Cypher's rules, or `None` when the value is for Python to judge.
/// `null` is never true, values of different kinds are unequal and unordered, and values of
/// the same kind compare as Python compares them, as `equals()` and `order()` do.
fn compare(
    value: &Bound<'_, PyAny>,
    op: Op,
    constant: &Bound<'_, PyAny>,
    class: Class,
) -> Option<bool> {
    match classify(value) {
        Class::Other => None,
        Class::Null => Some(false),
        kind if kind != class => Some(op == Op::Ne),
        _ => match op {
            Op::Eq | Op::Ne => {
                // as `equals()`, which counts a comparison that fails as unequal
                let same = value.eq(constant).unwrap_or(false);
                Some(same != (op == Op::Ne))
            }
            _ => value
                .rich_compare(constant, op.compare_op())
                .ok()?
                .is_truthy()
                .ok(),
        },
    }
}

pub(crate) fn run(
    graph: &Graph,
    py: Python,
    program: &Bound<'_, PyAny>,
    limit: Option<usize>,
    count_only: bool,
) -> PyResult<PyObject> {
    let program = parse_program(program)?;
    let mut search = Search {
        py,
        graph,
        program: &program,
        binding: vec![Val::Unset; program.kinds.len()],
        used_nodes: vec![FxHashSet::default(); program.stages.len()],
        used_edges: vec![FxHashSet::default(); program.stages.len()],
        found: vec![false; program.stages.len()],
        all_nodes: None,
        seeded: program
            .stages
            .iter()
            .map(|stage| vec![None; stage.steps.len()])
            .collect(),
        limit,
        count_only,
        count: 0,
        rows: Vec::new(),
        first: None,
        ticks: 0,
    };
    if limit != Some(0) {
        search.stage(0)?;
    }
    if count_only {
        let first = search.first.map_or_else(|| py.None(), |row| row.into_any());
        return Ok((search.count, first).into_pyobject(py)?.into_any().unbind());
    }
    Ok(PyList::new(py, search.rows)?.into_any().unbind())
}
