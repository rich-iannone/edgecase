"""Plots of the viewer's in Panel apps.

`GraphPane` shows a plot drawn with `backend="viewer"` (or a graph, which it plots so), with
its selection, view and filters as parameters, which follow the viewer, as it has them (see
`StepPlayer.selection`), and move it when set. This module needs Panel
(`pip install edgecase[panel]`).

```python
import edgecase as ec
import panel as pn
from edgecase.panel import GraphPane

pn.extension()
pane = GraphPane(ec.karate_club().plot(backend="viewer"))
picked = pn.bind(lambda s: f"Selected: {s['nodes']}", pane.param.selection)
pn.Column(pane, picked).servable()
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import panel as pn
import param

from edgecase.svg_renderer import STATES, _app_html, _app_plot, _app_state, _app_value

__all__ = ["GraphPane"]


class GraphPane(pn.custom.AnyWidgetComponent):
    """A plot of the viewer's in a Panel app, with its state as parameters.

    Its `object` is a plot drawn with `backend="viewer"`, or a graph, which is plotted so, with
    `seed=1`. A new one replaces the plot, and keeps the state. The state is the pane's, which
    starts as the plot's, and which the app doesn't change in the plot, as its sessions may share
    the plot.

    `selection`, `view` and `filters` follow the viewer, in the forms of `StepPlayer.selection`,
    `.view` and `.filters`, with lists for the edges' ends and the filters' ranges, and set it
    as they are set, in those forms. The view and the filters follow once they keep still for a
    moment, as a drag moves them.

    Parameters
    ----------
    object
        A plot drawn with `backend="viewer"`, or a graph.
    **params
        The pane's parameters, as `selection=`, `view=` and `filters=`, which the plot starts
        with, and Panel's, as `margin=`.

    Examples
    --------
    ```python
    import edgecase as ec
    import panel as pn
    from edgecase.panel import GraphPane

    pn.extension()
    pane = GraphPane(ec.karate_club(), selection={"nodes": [1, 34]})
    button = pn.widgets.Button(label="Clear")
    button.on_click(lambda event: setattr(pane, "selection", None))
    pn.Column(button, pane).servable()
    ```
    """

    object = param.Parameter(default=None, doc="The plot shown, or a graph to plot.")
    html = param.String(default="", doc="The plot's HTML, which the viewer shows.")
    selection = param.Dict(
        default={"nodes": [], "edges": []},
        allow_None=True,
        doc="The selected nodes' IDs and edges' ends, as `StepPlayer.selection` has them.",
    )
    view = param.Dict(
        default=None,
        allow_None=True,
        doc="The view's center and zoom, as `StepPlayer.view` has them.",
    )
    filters = param.List(
        default=[],
        allow_None=True,
        doc="The filters narrowed, as `StepPlayer.filters` has them.",
    )

    # The runtime, whose `render()` shows the plot (see the end of `static/viewer.js`)
    _esm = Path(__file__).parent / "static" / "viewer.js"
    _rename = {"object": None}

    def __init__(self, object: Any = None, **params: Any) -> None:
        plot = None if object is None else _app_plot(object, "GraphPane")
        start = {} if plot is None else _app_state(plot)
        # What a view given leaves out, as the plot has it
        home = start.get("view") or {}
        for name in STATES:
            if name in params:
                start[name] = params.pop(name)
        html = "" if plot is None else _app_html(plot)
        self._directed = plot is not None and plot._directed
        super().__init__(object=object, html=html, **params)
        self._fix(start, home)
        self.param.watch(self._drawn, ["object"])
        self.param.watch(self._moved, list(STATES))

    def _fix(self, state: dict, now: dict) -> None:
        fixed = {
            name: _app_value(name, state[name], now, self._directed)
            for name in STATES
            if name in state
        }
        if fixed:
            self.param.update(**fixed)

    def _drawn(self, event: Any) -> None:
        if event.new is None:
            self.html = ""
            return
        plot = _app_plot(event.new, "GraphPane")
        self._directed = plot._directed
        self.html = _app_html(plot)

    def _moved(self, event: Any) -> None:
        # As the viewer has it, which then holds it as given
        now = event.old if event.name == "view" else None
        fixed = _app_value(event.name, event.new, now, self._directed)
        if fixed != event.new:
            setattr(self, event.name, fixed)
