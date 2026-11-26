// What `highlight=` reads from the graph for an attribute's name: the nodes or edges whose
// value marks them, and the ends of the edges picked out, read here rather than node by node
// and edge by edge in Python
use crate::graph::Graph;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySet, PyTuple};

/// Whether an attribute value picks its node or edge out: any value but a missing one (None or
/// NaN) or False. A 0 counts, being, e.g., the first node of a path
fn is_marked(v: &Bound<PyAny>) -> bool {
    !(v.is_none()
        || v.downcast::<PyBool>().is_ok_and(|b| !b.is_true())
        || v.downcast::<PyFloat>().is_ok_and(|f| f.value().is_nan()))
}

/// The marked values of attribute `name` among `attrs`, or None if none of them has it
fn marked<'py, 'a, T: Copy>(
    py: Python<'py>,
    attrs: impl Iterator<Item = (T, Option<&'a Py<PyDict>>)> + Clone,
    name: &Bound<'py, PyAny>,
) -> PyResult<Option<Vec<(T, Bound<'py, PyAny>)>>> {
    let mut any = false;
    for (_, a) in attrs.clone() {
        if let Some(a) = a {
            if a.bind(py).contains(name)? {
                any = true;
                break;
            }
        }
    }
    if !any {
        return Ok(None);
    }
    let mut out = Vec::new();
    for (id, a) in attrs {
        if let Some(v) = a.map(|a| a.bind(py).get_item(name)).transpose()?.flatten() {
            if is_marked(&v) {
                out.push((id, v));
            }
        }
    }
    Ok(Some(out))
}

/// The nodes marked by node attribute `name`, in the order of the graph's node IDs, or None if
/// no node has it. When their values are two or more distinct whole numbers, as
/// `shortest_path_order` holds a path, the nodes in the order of the values as well
#[pyfunction]
pub fn marked_nodes(
    py: Python<'_>,
    graph: PyRef<'_, Graph>,
    name: Bound<'_, PyAny>,
) -> PyResult<Option<(Vec<u64>, Option<Vec<u64>>)>> {
    let ids = graph.get_node_ids();
    let attrs = ids.iter().map(|&id| {
        let node = graph.node_by_id(id);
        (id, node.and_then(|n| n.attributes.as_ref()))
    });
    let Some(found) = marked(py, attrs, &name)? else {
        return Ok(None);
    };
    let values: Vec<_> = found.iter().map(|(_, v)| v.clone()).collect();
    let whole = values
        .iter()
        .all(|v| v.is_instance_of::<PyInt>() && !v.is_instance_of::<PyBool>());
    let path = if values.len() >= 2 && whole && PySet::new(py, &values)?.len() == values.len() {
        let pairs = PyList::empty(py);
        for (id, v) in &found {
            pairs.append(PyTuple::new(
                py,
                [v.clone(), (*id).into_pyobject(py)?.into_any()],
            )?)?;
        }
        pairs.sort()?;
        let mut order = Vec::with_capacity(found.len());
        for pair in pairs.iter() {
            order.push(pair.get_item(1)?.extract()?);
        }
        Some(order)
    } else {
        None
    };
    Ok(Some((found.into_iter().map(|(id, _)| id).collect(), path)))
}

/// The edges marked by edge attribute `name`, in the order of the graph's edges, or None if no
/// edge has it
#[pyfunction]
pub fn marked_edges(
    py: Python<'_>,
    graph: PyRef<'_, Graph>,
    name: Bound<'_, PyAny>,
) -> PyResult<Option<Vec<u64>>> {
    let ids = graph.edge_ids();
    let attrs = ids.iter().map(|&id| {
        let edge = graph.edge_by_id(id);
        (id, edge.and_then(|e| e.attributes.as_ref()))
    });
    Ok(marked(py, attrs, &name)?.map(|found| found.into_iter().map(|(id, _)| id).collect()))
}

/// The ends of edges `edge_ids`
#[pyfunction]
pub fn edge_ends(graph: PyRef<'_, Graph>, edge_ids: Vec<u64>) -> Vec<u64> {
    edge_ids
        .into_iter()
        .filter_map(|id| graph.get_edge_by_id(id))
        .flat_map(|(source, target)| [source, target])
        .collect()
}
