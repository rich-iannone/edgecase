"""Plots of the viewer's in Streamlit apps.

`st_graph()` shows a plot drawn with `backend="viewer"` (or a graph, which it plots so) and
returns its selection, view and filters, as the viewer has them (see `StepPlayer.selection`). A
new selection or filters run the script again; the view is kept for them, and the plot keeps
all three as the script runs again. This module needs Streamlit
(`pip install edgecase[streamlit]`).

```python
import edgecase as ec
import streamlit as st
from edgecase.streamlit import st_graph

g = ec.karate_club()
state = st_graph(g.plot(backend="viewer"), key="club")
st.write("Selected:", state["selection"]["nodes"])
```
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, Optional

import streamlit as st

from edgecase.svg_renderer import (
    STATES,
    _app_html,
    _app_plot,
    _app_state,
    _app_value,
    _runtime_js,
)

__all__ = ["st_graph"]

_UNSET: Any = object()
_component: Any = None


def _mount() -> Any:
    # A component whose module is the runtime, which shows the plot (see `streamlit()` at the
    # end of `static/viewer.js`)
    global _component
    if _component is None:
        _component = st.components.v2.component("edgecase_graph", js=_runtime_js())
    return _component


def _nothing() -> None:
    pass


def st_graph(
    plot: Any,
    *,
    key: Optional[str] = None,
    selection: Any = _UNSET,
    view: Any = _UNSET,
    filters: Any = _UNSET,
    on_change: Optional[Callable[[], Any]] = None,
) -> Dict[str, Any]:
    """Show a plot of the viewer's in a Streamlit app, and get its state.

    The plot is drawn with `backend="viewer"`, or is a graph, which is plotted so, with
    `seed=1`. A plot of the force layout drawn without a seed is placed anew as the script runs
    again. Its state is the session's, which the plot keeps as the script runs again, by `key`,
    and a new plot of the same key starts with. A new selection or filters in the viewer run the
    script again, and a new view waits for them, to come back with them.

    What `selection=`, `view=` or `filters=` gives is set once, when it is first given and when
    it changes, and the viewer can change it after that. As the viewer takes it, the script
    runs again, which returns it.

    Parameters
    ----------
    plot
        A plot drawn with `backend="viewer"`, or a graph.
    key
        The plot's key, unique in the app, which keeps its state as the script runs again.
        Default is `None`, for one that comes of the plot, which keeps it while the plot is
        the same.
    selection
        The nodes and edges to select, as `StepPlayer.selection` takes them: a dict of
        `"nodes"`, IDs, and `"edges"`, `(u, v)` pairs. `None` clears the selection.
    view
        The view, as `StepPlayer.view` takes it: a dict of `"x"`, `"y"` and `"zoom"`, where a
        key left out stays as it is. `None` is the view that fits the graph.
    filters
        The filters, as `StepPlayer.filters` takes them: a list of dicts of `"attr"`, `"of"`
        and `"range"`. `None` or `[]` clears them.
    on_change
        A function to call, without arguments, before the script runs again for a new
        selection or filters.

    Returns
    -------
    dict
        The state, of `"selection"`, `"view"` and `"filters"`, in the forms of
        `StepPlayer.selection`, `.view` and `.filters`, with lists for the edges' ends and the
        filters' ranges, as of the viewer's last selection or filters.

    Raises
    ------
    TypeError, ValueError
        For a `plot` that isn't of the viewer's or a graph, or a state that isn't of its form.

    Examples
    --------
    ```python
    import edgecase as ec
    import streamlit as st
    from edgecase.streamlit import st_graph

    g = ec.karate_club()
    if st.button("Select the leaders"):
        st.session_state.pick = {"nodes": [1, 34]}
    state = st_graph(g, key="club", selection=st.session_state.get("pick"))
    st.write(state["selection"])
    ```
    """
    plot = _app_plot(plot, "st_graph")
    html = _app_html(plot)
    # The state that a new plot starts with, the plot's, with what is given, as the viewer has
    # it (where a view of `None` is the plot's, which fits the graph)
    start = _app_state(plot)
    given = {}
    for name, value in zip(STATES, (selection, view, filters)):
        if value is not _UNSET:
            given[name] = _app_value(name, value, directed=plot._directed)
            if given[name] is not None:
                start[name] = _app_value(name, value, start["view"], plot._directed)
    if key is None:
        key = "edgecase-" + hashlib.sha1(html.encode("utf-8")).hexdigest()[:12]
    result = _mount()(
        key=key,
        data={"html": html, "key": str(key), "state": start, "set": given},
        default={"state": start},
        on_state_change=on_change or _nothing,
    )
    held = result.get("state") or start
    return {name: _app_value(name, held.get(name)) for name in STATES}
