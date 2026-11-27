"""Plots of the viewer's in Solara apps.

`GraphView` shows a plot drawn with `backend="viewer"` (or a graph, which it plots so), with
its selection, view and filters as values or reactive variables, which follow the viewer, as it
has them (see `StepPlayer.selection`), and move it when set. This module needs Solara
(`pip install edgecase[solara]`).

```python
import edgecase as ec
import solara
from edgecase.solara import GraphView

plot = ec.karate_club().plot(backend="viewer")
selection = solara.reactive({"nodes": [1, 34], "edges": []})

@solara.component
def Page():
    GraphView(plot, selection=selection)
    solara.Text(f"Selected: {selection.value['nodes']}")
```
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import solara

from edgecase.svg_renderer import (
    STATES,
    _app_html,
    _app_plot,
    _app_state,
    _app_value,
    _widget_class,
)

__all__ = ["GraphView"]


@solara.component
def GraphView(
    plot: Any,
    selection: Any = None,
    view: Any = None,
    filters: Any = None,
    on_selection: Optional[Callable[[Any], Any]] = None,
    on_view: Optional[Callable[[Any], Any]] = None,
    on_filters: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """A plot of the viewer's in a Solara app, with its state as values or reactive variables.

    `plot` is a plot drawn with `backend="viewer"`, or a graph, which is plotted so, with
    `seed=1`. A new one replaces the plot, and keeps the state. The state is the component's,
    which starts as the plot's, and which the app doesn't change in the plot, as its sessions
    may share the plot.

    `selection`, `view` and `filters` are each a value, which the plot starts with and takes
    as it changes, or a reactive variable (`solara.reactive()`), which also follows the
    viewer, in the forms of `StepPlayer.selection`, `.view` and `.filters`, with lists for the
    edges' ends and the filters' ranges. `on_selection`, `on_view` and `on_filters` are called
    with them as they change. The view and the filters follow once they keep still for a
    moment, as a drag moves them.

    Parameters
    ----------
    plot
        A plot drawn with `backend="viewer"`, or a graph.
    selection
        The nodes and edges selected, as `StepPlayer.selection` has them. Default is `None`,
        for the plot's.
    view
        The view, as `StepPlayer.view` has it, where a key left out stays as it is. Default
        is `None`, for the plot's.
    filters
        The filters, as `StepPlayer.filters` has them. Default is `None`, for the plot's.
    on_selection, on_view, on_filters
        Functions to call with the selection, the view and the filters as they change.

    Examples
    --------
    ```python
    import edgecase as ec
    import solara
    from edgecase.solara import GraphView

    g = ec.karate_club()
    view = solara.reactive(None)

    @solara.component
    def Page():
        solara.Button("Zoom in", on_click=lambda: view.set({"zoom": 3}))
        GraphView(g, view=view)
        if view.value:
            solara.Text(f"Zoom: {view.value['zoom']:.1f}")
    ```
    """
    shown = solara.use_memo(lambda: _app_plot(plot, "GraphView"), [plot])
    html = solara.use_memo(lambda: _app_html(shown), [shown])
    start = solara.use_memo(lambda: _app_state(shown), [shown])

    def own(name: str, value: Any) -> Any:
        return start[name] if value is None else value

    held = {
        "selection": solara.use_reactive(own("selection", selection), on_selection),
        "view": solara.use_reactive(own("view", view), on_view),
        "filters": solara.use_reactive(own("filters", filters), on_filters),
    }

    # What the viewer gives, in the forms that Python has them
    def follow(name: str) -> Callable[[Any], None]:
        return lambda value: held[name].set(_app_value(name, value))

    state = {name: _app_value(name, held[name].value, directed=shown._directed) for name in STATES}
    # A view of `None`, the one that fits the graph, as the plot has it
    if state["view"] is None:
        state["view"] = start["view"]
    return _widget_class().element(
        html=html, **state, **{f"on_{name}": follow(name) for name in STATES}
    )
