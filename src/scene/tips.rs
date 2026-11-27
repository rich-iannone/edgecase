// The tooltips of the nodes and edges: their values, with the floats rounded, written as JSON.
// The JSON is what Python's `json.dumps(..., default=str)` writes, character for character, so
// that the values of attributes of any type show as they did when Python wrote them
use super::pyfmt::Repr;
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use std::fmt::Write;

/// A value in a tooltip
pub enum Cell<'py> {
    Text(String),
    Int(u64),
    Float(f64),
    /// An attribute's value, as it is held
    Obj(Bound<'py, PyAny>),
    /// The numbers of a vector, once rounded
    List(Vec<Cell<'py>>),
}

/// The tooltips of a set of nodes or edges, a row of `(key, value)` entries for each, with the
/// keys in the order first seen
pub struct Tips<'py> {
    keys: Vec<Bound<'py, PyAny>>,
    // Each key's place in `keys`
    index: Bound<'py, PyDict>,
    rows: Vec<Vec<(usize, Cell<'py>)>>,
}

/// Nested deeper than this, a value is left to Python's `json`, which finds a circular one
const DEPTH_MAX: usize = 64;

/// `v` to `digits` decimal places, as Python's `round(v, digits)` rounds it
fn round_to(v: f64, digits: i64) -> f64 {
    // Past 323 places, Python leaves a float as it is, as do the digits written here
    if !v.is_finite() || digits > 323 {
        return v;
    }
    format!("{:.*}", digits as usize, v)
        .parse()
        .expect("a formatted float reads back")
}

/// The decimal places that keep 4 significant digits of `v` and all of its whole-number digits
fn sig_digits(v: f64) -> i64 {
    (3 - v.abs().log10().floor() as i64).max(0)
}

/// `v` to 4 significant digits, but keeping all of its whole-number digits
fn round_sig(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() {
        v
    } else {
        round_to(v, sig_digits(v))
    }
}

/// The numbers of `v`, if it's a non-empty list or tuple of nothing else (booleans aside)
fn num_vector<'py>(v: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
    let items: Vec<_> = if let Ok(list) = v.downcast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = v.downcast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        return None;
    };
    let numeric = |c: &Bound<PyAny>| {
        !c.is_instance_of::<PyBool>()
            && (c.is_instance_of::<PyInt>() || c.is_instance_of::<PyFloat>())
    };
    (!items.is_empty() && items.iter().all(numeric)).then_some(items)
}

/// Python's `round(v, digits)`, for floats of other types (such as NumPy's)
fn py_round<'py>(v: &Bound<'py, PyAny>, digits: i64) -> PyResult<Bound<'py, PyAny>> {
    let py = v.py();
    py.import(intern!(py, "builtins"))?
        .getattr(intern!(py, "round"))?
        .call1((v, digits))
}

/// `s` as a JSON string, non-ASCII characters escaped
fn write_text(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0; 2];
                for unit in c.encode_utf16(&mut units) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
    }
    out.push('"');
}

/// A float as JSON. As text, one that JSON can't hold (NaN or an infinity) is written as a
/// string, as `str()` writes it
fn write_float(out: &mut String, v: f64, as_text: bool) {
    if v.is_finite() {
        let _ = write!(out, "{}", Repr(v));
    } else if as_text {
        write_text(out, &Repr(v).to_string());
    } else {
        out.push_str(if v.is_nan() {
            "NaN"
        } else if v > 0.0 {
            "Infinity"
        } else {
            "-Infinity"
        });
    }
}

/// Python's `json.dumps(v, default=str)`, for what is left to it
fn python_json(out: &mut String, v: &Bound<PyAny>) -> PyResult<()> {
    let py = v.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "default"), py.get_type::<PyString>())?;
    let text = py
        .import(intern!(py, "json"))?
        .getattr(intern!(py, "dumps"))?
        .call((v,), Some(&kwargs))?;
    out.push_str(text.downcast::<PyString>()?.to_str()?);
    Ok(())
}

/// An int as JSON: its digits
fn write_int(out: &mut String, v: &Bound<PyInt>) -> PyResult<()> {
    match v.extract::<i64>() {
        Ok(i) => {
            let _ = write!(out, "{i}");
        }
        // As int's own repr() writes it, however large (and whatever the type's repr())
        Err(_) => {
            let py = v.py();
            let digits = py
                .get_type::<PyInt>()
                .getattr(intern!(py, "__repr__"))?
                .call1((v,))?;
            out.push_str(digits.downcast::<PyString>()?.to_str()?);
        }
    }
    Ok(())
}

/// A dict key as JSON, which is always a string
fn write_key(out: &mut String, key: &Bound<PyAny>) -> PyResult<()> {
    if let Ok(s) = key.downcast::<PyString>() {
        match s.to_str() {
            Ok(s) => write_text(out, s),
            Err(_) => python_json(out, key)?,
        }
    } else if let Ok(f) = key.downcast::<PyFloat>() {
        let mut text = String::new();
        write_float(&mut text, f.value(), false);
        write_text(out, &text);
    } else if let Ok(b) = key.downcast::<PyBool>() {
        write_text(out, if b.is_true() { "true" } else { "false" });
    } else if key.is_none() {
        write_text(out, "null");
    } else if let Ok(i) = key.downcast::<PyInt>() {
        let mut text = String::new();
        write_int(&mut text, i)?;
        write_text(out, &text);
    } else {
        // Python's json raises its TypeError for any other key
        let dict = PyDict::new(key.py());
        dict.set_item(key, key.py().None())?;
        python_json(out, &dict.into_any())?;
        return Err(PyTypeError::new_err(
            "keys must be str, int, float, bool or None",
        ));
    }
    Ok(())
}

/// Any value as JSON, `str()` of it standing for one that JSON has no type for. `as_text`
/// writes floats that JSON can't hold as text, in this value and in the lists in it
fn write_obj(out: &mut String, v: &Bound<PyAny>, as_text: bool, depth: usize) -> PyResult<()> {
    if depth > DEPTH_MAX {
        return python_json(out, v);
    }
    if v.is_none() {
        out.push_str("null");
    } else if let Ok(b) = v.downcast::<PyBool>() {
        out.push_str(if b.is_true() { "true" } else { "false" });
    } else if let Ok(s) = v.downcast::<PyString>() {
        match s.to_str() {
            Ok(s) => write_text(out, s),
            // A lone surrogate, which json escapes
            Err(_) => python_json(out, v)?,
        }
    } else if let Ok(i) = v.downcast::<PyInt>() {
        write_int(out, i)?;
    } else if let Ok(f) = v.downcast::<PyFloat>() {
        if as_text && !f.value().is_finite() && !v.is_exact_instance_of::<PyFloat>() {
            write_text(out, v.str()?.to_str()?);
        } else {
            write_float(out, f.value(), as_text);
        }
    } else if v.is_instance_of::<PyList>() || v.is_instance_of::<PyTuple>() {
        out.push('[');
        for (k, item) in v.try_iter()?.enumerate() {
            if k > 0 {
                out.push_str(", ");
            }
            write_obj(out, &item?, as_text, depth + 1)?;
        }
        out.push(']');
    } else if let Ok(dict) = v.downcast_exact::<PyDict>() {
        out.push('{');
        for (k, (key, value)) in dict.iter().enumerate() {
            if k > 0 {
                out.push_str(", ");
            }
            write_key(out, &key)?;
            out.push_str(": ");
            write_obj(out, &value, false, depth + 1)?;
        }
        out.push('}');
    } else if v.is_instance_of::<PyDict>() {
        python_json(out, v)?;
    } else {
        write_obj(out, &v.str()?.into_any(), false, depth + 1)?;
    }
    Ok(())
}

fn write_cell(out: &mut String, cell: &Cell, as_text: bool) -> PyResult<()> {
    match cell {
        Cell::Text(s) => write_text(out, s),
        Cell::Int(i) => {
            let _ = write!(out, "{i}");
        }
        Cell::Float(v) => write_float(out, *v, as_text),
        Cell::Obj(v) => write_obj(out, v, as_text, 0)?,
        Cell::List(cells) => {
            out.push('[');
            for (k, c) in cells.iter().enumerate() {
                if k > 0 {
                    out.push_str(", ");
                }
                write_cell(out, c, as_text)?;
            }
            out.push(']');
        }
    }
    Ok(())
}

/// JSON safe in a `data-attrs='...'` attribute: `&`, `'` and `<` as character references
fn attribute_safe(json: String) -> String {
    if !json.contains(['&', '\'', '<']) {
        return json;
    }
    let mut out = String::with_capacity(json.len() + 16);
    for c in json.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '\'' => out.push_str("&#39;"),
            '<' => out.push_str("&lt;"),
            _ => out.push(c),
        }
    }
    out
}

impl<'py> Tips<'py> {
    pub fn new(py: Python<'py>) -> Self {
        Tips {
            keys: Vec::new(),
            index: PyDict::new(py),
            rows: Vec::new(),
        }
    }

    /// `key`'s place among the keys
    pub fn key(&mut self, key: &Bound<'py, PyAny>) -> PyResult<usize> {
        if let Some(k) = self.index.get_item(key)? {
            return k.extract();
        }
        self.index.set_item(key, self.keys.len())?;
        self.keys.push(key.clone());
        Ok(self.keys.len() - 1)
    }

    /// A row: `head`, then `attrs`, whose values replace those of the head's keys
    pub fn push(
        &mut self,
        mut row: Vec<(usize, Cell<'py>)>,
        attrs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<()> {
        let head = row.len();
        if let Some(attrs) = attrs {
            for (key, value) in attrs.iter() {
                let k = self.key(&key)?;
                match row[..head].iter_mut().find(|(j, _)| *j == k) {
                    Some(entry) => entry.1 = Cell::Obj(value),
                    None => row.push((k, Cell::Obj(value))),
                }
            }
        }
        self.rows.push(row);
        Ok(())
    }

    /// Round the floats, which can run to 16 digits. A single number keeps 4 significant
    /// digits. The numbers in a vector (e.g., a stored `pos`) are rounded to the same decimal
    /// place in every row: a hundredth of the typical spacing of that many points spread over
    /// the vectors' range. Significant digits would instead show a coordinate that is nearly 0
    /// as noise like 1.3e-17
    pub fn round(&mut self) -> PyResult<()> {
        let n = self.keys.len();
        // For each key, the vectors, and the least and greatest of their finite numbers
        let mut vectors = vec![0usize; n];
        let mut range: Vec<Option<(f64, f64)>> = vec![None; n];
        for row in &self.rows {
            for (k, cell) in row {
                let Cell::Obj(v) = cell else { continue };
                let Some(items) = num_vector(v) else { continue };
                vectors[*k] += 1;
                for c in items {
                    let c: f64 = c.extract()?;
                    if c.is_finite() {
                        range[*k] = Some(match range[*k] {
                            None => (c, c),
                            Some((lo, hi)) => (lo.min(c), hi.max(c)),
                        });
                    }
                }
            }
        }
        let decimals: Vec<Option<i64>> = (0..n)
            .map(|k| {
                range[k].map(|(lo, hi)| {
                    let spread = hi - lo;
                    if spread > 0.0 {
                        (((vectors[k] as f64).sqrt() / spread).log10().ceil() as i64)
                            .saturating_add(2)
                            .max(0)
                    } else {
                        2
                    }
                })
            })
            .collect();
        for row in &mut self.rows {
            for (k, cell) in row.iter_mut() {
                let tidied = match cell {
                    Cell::Float(v) => Cell::Float(round_sig(*v)),
                    Cell::Obj(v) => {
                        if let Ok(f) = v.downcast::<PyFloat>() {
                            let x = f.value();
                            if v.is_exact_instance_of::<PyFloat>() {
                                Cell::Float(round_sig(x))
                            } else if x == 0.0 || !x.is_finite() {
                                continue;
                            } else {
                                Cell::Obj(py_round(v, sig_digits(x))?)
                            }
                        } else if let (Some(d), Some(items)) = (decimals[*k], num_vector(v)) {
                            let mut out = Vec::with_capacity(items.len());
                            for c in items {
                                out.push(if c.is_exact_instance_of::<PyFloat>() {
                                    Cell::Float(round_to(c.downcast::<PyFloat>()?.value(), d))
                                } else if c.is_instance_of::<PyFloat>() {
                                    Cell::Obj(py_round(&c, d)?)
                                } else {
                                    Cell::Obj(c)
                                });
                            }
                            Cell::List(out)
                        } else {
                            continue;
                        }
                    }
                    _ => continue,
                };
                *cell = tidied;
            }
        }
        Ok(())
    }

    /// Each row as JSON for a `data-attrs='...'` attribute
    pub fn attrs_json(&self) -> PyResult<Vec<String>> {
        let mut keys = Vec::with_capacity(self.keys.len());
        for key in &self.keys {
            let mut text = String::new();
            write_key(&mut text, key)?;
            keys.push(text);
        }
        self.rows
            .iter()
            .map(|row| {
                let mut out = String::from("{");
                for (j, (k, cell)) in row.iter().enumerate() {
                    if j > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&keys[*k]);
                    out.push_str(": ");
                    write_cell(&mut out, cell, false)?;
                }
                out.push('}');
                Ok(attribute_safe(out))
            })
            .collect()
    }

    /// The rows as one JSON table for a `<script type="application/json">`: the keys, and a
    /// column of values for each. A column that every row has a value in is a list, and any
    /// other an object from row number to value, so that either is read as `column[i]`. Numbers
    /// that JSON can't hold (NaN, infinities) are written as text, since one would keep the
    /// whole table from being read. With `order`, row `i` of the table is row `order[i]`
    pub fn table(&self, order: Option<&[usize]>) -> PyResult<String> {
        let n = self.keys.len();
        let mut columns: Vec<Vec<(usize, &Cell)>> = (0..n).map(|_| Vec::new()).collect();
        for i in 0..self.rows.len() {
            for (k, cell) in &self.rows[order.map_or(i, |o| o[i])] {
                columns[*k].push((i, cell));
            }
        }
        let mut out = String::from("{\"k\": [");
        for (k, key) in self.keys.iter().enumerate() {
            if k > 0 {
                out.push_str(", ");
            }
            write_obj(&mut out, key, false, 0)?;
        }
        out.push_str("], \"c\": [");
        for (k, column) in columns.iter().enumerate() {
            if k > 0 {
                out.push_str(", ");
            }
            let full = column.len() == self.rows.len();
            out.push(if full { '[' } else { '{' });
            for (j, (i, cell)) in column.iter().enumerate() {
                if j > 0 {
                    out.push_str(", ");
                }
                if !full {
                    let _ = write!(out, "\"{i}\": ");
                }
                write_cell(&mut out, cell, true)?;
            }
            out.push(if full { ']' } else { '}' });
        }
        out.push_str("]}");
        // "<" written as an escape, so that no "</script>" in a value can end the script
        Ok(out.replace('<', "\\u003c"))
    }
}

/// Rows of `rows`, dicts, as tooltips
fn rows_of<'py>(py: Python<'py>, rows: Vec<Bound<'py, PyDict>>) -> PyResult<Tips<'py>> {
    let mut tips = Tips::new(py);
    for row in &rows {
        tips.push(Vec::new(), Some(row))?;
    }
    tips.round()?;
    Ok(tips)
}

/// `rows` (dicts) as `plot()` writes the tooltips of nodes and edges, its floats rounded: as
/// JSON for a `data-attrs='...'` attribute each
#[pyfunction]
pub fn tooltip_attrs(py: Python<'_>, rows: Vec<Bound<'_, PyDict>>) -> PyResult<Vec<String>> {
    rows_of(py, rows)?.attrs_json()
}

/// `rows` (dicts) as `plot()` writes the tooltips of edges in lean SVG, its floats rounded: as
/// one JSON table, `{"k": keys, "c": columns}`
#[pyfunction]
pub fn tooltip_table(py: Python<'_>, rows: Vec<Bound<'_, PyDict>>) -> PyResult<String> {
    rows_of(py, rows)?.table(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_are_rounded_as_python_rounds_them() {
        // From Python's round() and the renderer's _round_sig()
        for (v, digits, rounded) in [
            (0.125, 2, 0.12),
            (0.375, 2, 0.38),
            (2.5, 0, 2.0),
            (-1e-17, 2, -0.0),
            (5e-324, 400, 5e-324),
            (1.23456789, 3, 1.235),
        ] {
            assert_eq!(round_to(v, digits).to_bits(), f64::to_bits(rounded), "{v}");
        }
        for (v, rounded) in [
            (1.2345678, 1.235),
            (12345.678, 12346.0),
            (0.000123456, 0.0001235),
            (-0.0, -0.0),
            (f64::INFINITY, f64::INFINITY),
        ] {
            assert_eq!(round_sig(v).to_bits(), f64::to_bits(rounded), "{v}");
        }
    }

    #[test]
    fn text_is_escaped_as_json_escapes_it() {
        // As json.dumps() writes it
        let mut out = String::new();
        write_text(&mut out, "a\"b\\c\nd\te\u{1}f\u{7f}é😀\u{8}\u{c} <&'");
        assert_eq!(
            out,
            r#""a\"b\\c\nd\te\u0001f\u007f\u00e9\ud83d\ude00\b\f <&'""#
        );
        assert_eq!(
            attribute_safe(out),
            r#""a\"b\\c\nd\te\u0001f\u007f\u00e9\ud83d\ude00\b\f &lt;&amp;&#39;""#
        );
    }
}
