"""Plots of the viewer's in Shiny apps.

`render_graph` renders a plot drawn with `backend="viewer"` (or a graph, which it plots so) to
an output, which `output_graph()` places in the app's UI. The plot's selection, view and filters
are the inputs `<id>_selection`, `<id>_view` and `<id>_filters`, of each session, as the viewer
has them (see `StepPlayer.selection`), and `update_graph()` sets them. This module needs Shiny
(`pip install edgecase[shiny]`).

```python
import edgecase as ec
from edgecase.shiny import output_graph, render_graph, update_graph
from shiny import App, reactive, render, ui

g = ec.karate_club()
app_ui = ui.page_fluid(
    ui.input_action_button("leaders", "Select the leaders"),
    output_graph("club"),
    ui.output_text("picked"),
)

def server(input, output, session):
    @render_graph
    def club():
        return g.plot(backend="viewer")

    @render.text
    def picked():
        return f"Selected: {input.club_selection()['nodes']}"

    @reactive.effect
    @reactive.event(input.leaders)
    def _():
        update_graph("club", selection={"nodes": [1, 34]})

app = App(app_ui, server)
```
"""

from __future__ import annotations

from typing import Any, Optional

from htmltools import HTMLDependency, Tag, div
from shiny.input_handler import input_handlers
from shiny.module import resolve_id
from shiny.render.renderer import Renderer
from shiny.session import Session, require_active_session

from edgecase.svg_renderer import STATES, _app_html, _app_plot, _app_state, _app_value

__all__ = ["output_graph", "render_graph", "update_graph"]

_UNSET: Any = object()


def _dependency() -> HTMLDependency:
    # The runtime, which also binds the outputs (see the end of `static/viewer.js`)
    from edgecase import __version__

    return HTMLDependency(
        "edgecase-viewer",
        __version__,
        source={"package": "edgecase", "subdir": "static"},
        script={"src": "viewer.js", "type": "module"},
    )


def output_graph(id: str) -> Tag:
    """The place of a plot of the viewer's in a Shiny app's UI.

    `render_graph` renders the plot to it, as the plot's size. The page loads the viewer's
    runtime once, for all of its plots.

    Parameters
    ----------
    id
        The output's ID, the name of the function that `render_graph` decorates. The plot's
        state is the inputs of this ID with `_selection`, `_view` and `_filters`.

    Returns
    -------
    Tag
        A `<div>` for the plot.

    Examples
    --------
    ```python
    from edgecase.shiny import output_graph
    from shiny import ui

    ui.page_fluid(output_graph("club"))
    ```
    """
    return div(_dependency(), id=resolve_id(id), class_="ec-graph-output")


class render_graph(Renderer[Any]):
    """Render a plot of the viewer's to a Shiny output.

    The function it decorates returns a plot drawn with `backend="viewer"`, or a graph, which
    is plotted so, with `seed=1`, to be placed the same each time. Its output, of the function's
    name, shows in the UI where `output_graph()` places it, or where the function is, in Shiny
    Express. A plot of the force layout drawn without a seed is placed anew each time.

    The plot's state is each session's, in the inputs of the output's ID with `_selection`,
    `_view` and `_filters`, which follow the viewer, in the forms of `StepPlayer.selection`,
    `.view` and `.filters`, and `update_graph()` sets. A plot rendered again starts with the
    plot's own state, which the app doesn't change, as its sessions may share the plot. A
    function that returns `None` shows nothing.

    Examples
    --------
    ```python
    import edgecase as ec
    from edgecase.shiny import render_graph
    from shiny import render

    g = ec.karate_club()

    def server(input, output, session):
        @render_graph
        def club():
            return g.plot(backend="viewer")

        @render.text
        def zoom():
            return f"Zoom: {input.club_view()['zoom']:.1f}"
    ```
    """

    def auto_output_ui(self) -> Tag:
        return output_graph(self.output_id)

    async def transform(self, value: Any) -> Any:
        plot = _app_plot(value, "render_graph")
        return {"html": _app_html(plot), "state": _app_state(plot)}


def update_graph(
    id: str,
    *,
    selection: Any = _UNSET,
    view: Any = _UNSET,
    filters: Any = _UNSET,
    session: Optional[Session] = None,
) -> None:
    """Set the state of a plot that `render_graph` renders, in a session of a Shiny app.

    What is given is set in the viewer, and its inputs follow, once the session's outputs are
    sent, so a plot rendered in the same flush takes it. What is left out stays as it is.

    Parameters
    ----------
    id
        The output's ID.
    selection
        The nodes and edges to select, as `StepPlayer.selection` takes them: a dict of
        `"nodes"`, IDs, and `"edges"`, `(u, v)` pairs. `None` clears the selection.
    view
        The view, as `StepPlayer.view` takes it: a dict of `"x"`, `"y"` and `"zoom"`, where a
        key left out stays as it is. `None` is the view that fits the graph.
    filters
        The filters, as `StepPlayer.filters` takes them: a list of dicts of `"attr"`, `"of"`
        and `"range"`. `None` or `[]` clears them.
    session
        The session. Default is `None`, for the current one.

    Raises
    ------
    TypeError, ValueError
        For a state that isn't of its form.

    Examples
    --------
    ```python
    from edgecase.shiny import update_graph
    from shiny import reactive

    def server(input, output, session):
        @reactive.effect
        @reactive.event(input.reset)
        def _():
            update_graph("club", selection=None, view=None, filters=None)
    ```
    """
    session = require_active_session(session)
    message: dict = {"id": session.ns(id)}
    for name, value in zip(STATES, (selection, view, filters)):
        if value is not _UNSET:
            message[name] = _app_value(name, value)

    async def send() -> None:
        await session.send_custom_message("edgecase-graph", message)

    session.on_flushed(send, once=True)


# The inputs, as the viewer gives them (see `tell()` at the end of `static/viewer.js`), in the
# forms that Python has them
def _input(name: str) -> Any:
    def handle(value: Any, input_name: Any, session: Any) -> Any:
        return _app_value(name, value)

    return handle


for _name in STATES:
    input_handlers.add(f"edgecase.{_name}", force=True)(_input(_name))
