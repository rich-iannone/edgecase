"""The plots of the viewer's in apps: Shiny, Streamlit, Panel and Solara.

Each framework's tests skip without it (`pip install edgecase[test]` has them all).
"""

import asyncio
import json
import re

import edgecase as ec
import edgecase.svg_renderer as svg_renderer
import pytest


def _plot():
    return ec.karate_club().plot(backend="viewer", seed=1)


def test_an_apps_plot_is_of_the_viewer():
    plot = _plot()
    assert svg_renderer._app_plot(plot, "f") is plot
    # A graph is plotted with the viewer, and the same each time
    g = ec.karate_club()
    drawn = svg_renderer._app_plot(g, "f")
    assert drawn._viewer
    assert svg_renderer._app_html(drawn) == svg_renderer._app_html(svg_renderer._app_plot(g, "f"))
    with pytest.raises(TypeError, match="f takes a plot of the viewer's or a graph, got int"):
        svg_renderer._app_plot(1, "f")
    with pytest.raises(ValueError, match="f takes a plot with backend='viewer'"):
        svg_renderer._app_plot(ec.karate_club().plot(seed=1), "f")


def test_an_apps_plot_has_the_same_html_when_drawn_again():
    one, two = _plot(), _plot()
    assert one._html != two._html
    html = svg_renderer._app_html(one)
    assert html == svg_renderer._app_html(two)
    assert re.search(r'<ec-graph id="ec-viewer-[0-9a-f]{12}"', html)
    assert "viewer.js" not in html and "customElements.define" not in html
    other = ec.karate_club().plot(backend="viewer", seed=2)
    assert svg_renderer._app_html(other) != html


def test_an_apps_state_is_as_the_viewer_has_it():
    plot = _plot()
    plot.selection = {"nodes": [1]}
    state = svg_renderer._app_state(plot)
    assert state == {
        "selection": {"nodes": [1], "edges": []},
        "view": plot.view,
        "filters": [],
    }
    value = svg_renderer._app_value
    assert value("selection", {"nodes": [3, 1], "edges": [(2, 1)]}, directed=False) == {
        "nodes": [1, 3],
        "edges": [[1, 2]],
    }
    assert value("selection", {"edges": [(2, 1)]})["edges"] == [[2, 1]]
    assert value("selection", None) == {"nodes": [], "edges": []}
    # A view leaves out what isn't given, or keeps it as it is now; None is the one that fits
    assert value("view", None) is None
    assert value("view", {"zoom": 3}) == {"zoom": 3.0}
    assert value("view", {"zoom": 3}, {"x": 1, "y": 2, "zoom": 1}) == {
        "x": 1.0,
        "y": 2.0,
        "zoom": 3.0,
    }
    assert value("view", {"zoom": 1e9})["zoom"] == svg_renderer._ZOOM_MAX
    with pytest.raises(ValueError, match="more than 0"):
        value("view", {"zoom": 0})
    assert value("filters", None) == []
    assert value("filters", [{"attr": "age", "of": "nodes", "range": (30, 40)}]) == [
        {"attr": "age", "of": "nodes", "range": [30, 40]}
    ]
    with pytest.raises(TypeError, match="filters takes a list of dicts"):
        value("filters", {"attr": "age"})


def test_the_runtime_shows_the_plot_in_each_framework():
    js = svg_renderer._runtime_js()
    # A Streamlit component, whose state the session keeps, the view without a rerun
    assert "function streamlit({ data, parentElement, setStateValue })" in js
    assert 'setStateValue("state", { ...app.state })' in js
    # Shiny's outputs and messages, whose inputs are of the output's ID
    assert 'shiny.outputBindings.register(new GraphOutput(), "edgecase.graph")' in js
    assert 'shiny.addCustomMessageHandler("edgecase-graph"' in js
    assert "`${el.id}_${name}:edgecase.${name}`" in js


# Shiny


class _Session:
    """A Shiny session, as `update_graph()` uses it, in a module of `prefix`."""

    def __init__(self, prefix=""):
        self.prefix, self.flushed, self.sent = prefix, [], []

    def ns(self, id):
        return f"{self.prefix}-{id}" if self.prefix else id

    def on_flushed(self, fn, once=False):
        assert once
        self.flushed.append(fn)

    async def send_custom_message(self, type, message):
        self.sent.append((type, message))

    def flush(self):
        for fn in self.flushed:
            asyncio.run(fn())


def test_shinys_output_has_the_runtime():
    pytest.importorskip("shiny")
    from edgecase import __version__
    from edgecase.shiny import output_graph

    tag = output_graph("club")
    assert tag.attrs == {"id": "club", "class": "ec-graph-output"}
    dep = tag.get_dependencies()[0]
    assert dep.name == "edgecase-viewer" and str(dep.version) == __version__
    assert dep.script == [{"src": "viewer.js", "type": "module"}]
    assert dep.source == {"package": "edgecase", "subdir": "static"}
    assert (svg_renderer.Path(svg_renderer.__file__).parent / "static" / "viewer.js").exists()


def test_shiny_renders_the_plot_with_its_state():
    pytest.importorskip("shiny")
    from edgecase.shiny import render_graph

    plot = _plot()
    plot.selection = {"nodes": [5]}

    @render_graph
    def club():
        return plot

    assert club.auto_output_ui().attrs["class"] == "ec-graph-output"
    data = asyncio.run(club.transform(plot))
    assert data == {"html": svg_renderer._app_html(plot), "state": svg_renderer._app_state(plot)}
    assert data["state"]["selection"] == {"nodes": [5], "edges": []}
    json.dumps(data)
    # A graph, plotted with the viewer
    drawn = asyncio.run(club.transform(ec.karate_club()))
    assert '<ec-graph id="ec-viewer-' in drawn["html"]
    with pytest.raises(TypeError, match="render_graph takes a plot of the viewer's"):
        asyncio.run(club.transform("club"))


def test_shiny_updates_the_plot_once_the_outputs_are_sent():
    pytest.importorskip("shiny")
    from edgecase.shiny import update_graph

    session = _Session()
    update_graph("club", selection={"nodes": [34, 1], "edges": [(2, 1)]}, session=session)
    assert session.sent == []
    session.flush()
    assert session.sent == [
        ("edgecase-graph", {"id": "club", "selection": {"nodes": [1, 34], "edges": [[2, 1]]}})
    ]
    # In a module, with what is left out not sent, and a view that fits the graph
    session = _Session("mod")
    update_graph("club", view={"zoom": 3}, filters=None, session=session)
    update_graph("club", view=None, session=session)
    session.flush()
    assert session.sent == [
        ("edgecase-graph", {"id": "mod-club", "view": {"zoom": 3.0}, "filters": []}),
        ("edgecase-graph", {"id": "mod-club", "view": None}),
    ]
    with pytest.raises(ValueError, match="more than 0"):
        update_graph("club", view={"zoom": -1}, session=session)
    with pytest.raises(RuntimeError, match="active Shiny session"):
        update_graph("club", selection=None)


def test_shinys_inputs_are_as_python_has_them():
    pytest.importorskip("shiny")
    import edgecase.shiny  # noqa: F401
    from shiny.input_handler import input_handlers

    def given(name, value):
        return input_handlers._process_value(f"edgecase.{name}", value, "club_" + name, None)

    assert given("selection", {"nodes": [1, 34], "edges": [[1, 2]]}) == {
        "nodes": [1, 34],
        "edges": [[1, 2]],
    }
    assert given("view", {"x": 1, "y": 2, "zoom": 3}) == {"x": 1.0, "y": 2.0, "zoom": 3.0}
    assert given("filters", [{"attr": "age", "of": "nodes", "range": [30, 40]}]) == [
        {"attr": "age", "of": "nodes", "range": [30, 40]}
    ]


# Streamlit


def _streamlit_app():
    import edgecase as ec
    import streamlit as st
    from edgecase.streamlit import st_graph

    g = ec.karate_club()
    st.session_state.state = st_graph(
        g, key="club", selection={"nodes": [34, 1, 1]}, view={"zoom": 2}
    )
    st.session_state.again = st_graph(g.plot(backend="viewer", seed=1))


def test_streamlit_shows_the_plot_and_returns_its_state():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_streamlit_app).run()
    assert not at.exception
    plot = ec.karate_club().plot(backend="viewer", seed=1)
    home = plot.view
    # What is given, as the viewer has it, and the plot's for the rest
    assert at.session_state.state == {
        "selection": {"nodes": [1, 34], "edges": []},
        "view": {**home, "zoom": 2.0},
        "filters": [],
    }
    assert at.session_state.again == svg_renderer._app_state(plot)
    shown = [e for e in at.main if type(e).__name__ == "UnknownElement"]
    assert [e.proto.component_name for e in shown] == ["edgecase_graph"] * 2
    assert shown[0].proto.js_content == svg_renderer._runtime_js()
    data = json.loads(shown[0].proto.json)
    assert data["html"] == svg_renderer._app_html(plot) and data["key"] == "club"
    assert data["set"] == {"selection": {"nodes": [1, 34], "edges": []}, "view": {"zoom": 2.0}}
    assert data["state"] == at.session_state.state
    # The key comes of the plot, without one
    again = json.loads(shown[1].proto.json)
    assert re.fullmatch("edgecase-[0-9a-f]{12}", again["key"]) and again["set"] == {}
    # The session keeps the state, which comes back as the script runs again
    assert at.session_state["club"] == {"state": at.session_state.state}
    at.run()
    assert at.session_state.state["selection"] == {"nodes": [1, 34], "edges": []}


# Panel


def test_panels_pane_has_the_state_as_parameters():
    pytest.importorskip("panel")
    from edgecase.panel import GraphPane

    plot = _plot()
    pane = GraphPane(plot, selection={"nodes": [34, 1]}, view={"zoom": 2})
    assert pane.html == svg_renderer._app_html(plot) and "object" in pane._rename
    assert pane.selection == {"nodes": [1, 34], "edges": []}
    assert pane.view == {**plot.view, "zoom": 2.0} and pane.filters == []
    # The plot's own state isn't the pane's
    assert plot.selection == {"nodes": [], "edges": []}
    # Set, as the viewer has it, and a view keeping what is left out
    pane.selection = {"edges": [(2, 1)]}
    assert pane.selection == {"nodes": [], "edges": [[1, 2]]}
    pane.selection = None
    assert pane.selection == {"nodes": [], "edges": []}
    pane.view = {"x": 10}
    assert pane.view == {**plot.view, "x": 10.0, "zoom": 2.0}
    pane.view = None
    assert pane.view is None
    pane.filters = [{"attr": "age", "of": "nodes", "range": (30, 40)}]
    assert pane.filters == [{"attr": "age", "of": "nodes", "range": [30, 40]}]
    # A new plot, which keeps the state
    pane.object = ec.karate_club().plot(backend="viewer", seed=2, layout="circular")
    assert pane.html == svg_renderer._app_html(pane.object) and pane.html != svg_renderer._app_html(
        plot
    )
    assert pane.filters == [{"attr": "age", "of": "nodes", "range": [30, 40]}]
    pane.object = None
    assert pane.html == ""
    assert GraphPane().html == ""
    with pytest.raises(ValueError, match="GraphPane takes a plot with backend='viewer'"):
        GraphPane(ec.karate_club().plot(seed=1))


# Solara


def test_solaras_view_has_the_state_as_reactive_variables():
    solara = pytest.importorskip("solara")
    from edgecase.solara import GraphView

    plot = _plot()
    selection = solara.reactive({"nodes": [34, 1]})
    view = solara.reactive(None)
    filters = []
    seen = []

    @solara.component
    def Page():
        GraphView(plot, selection=selection, view=view, on_filters=seen.append)

    box, rc = solara.render(Page(), handle_error=False)
    widget = rc.find(svg_renderer._widget_class()).widget
    assert widget.html == svg_renderer._app_html(plot)
    assert widget.selection == {"nodes": [1, 34], "edges": []}
    # A view of None, the one that fits the graph
    assert widget.view == plot.view and widget.filters == filters
    # The viewer's state, followed
    widget.selection = {"nodes": [5], "edges": [[2, 1]]}
    assert selection.value == {"nodes": [5], "edges": [[2, 1]]}
    widget.view = {"x": 1, "y": 2, "zoom": 3}
    assert view.value == {"x": 1.0, "y": 2.0, "zoom": 3.0}
    widget.filters = [{"attr": "age", "of": "nodes", "range": [30, 40]}]
    assert seen == [[{"attr": "age", "of": "nodes", "range": [30, 40]}]]
    # And set
    selection.set(None)
    assert widget.selection == {"nodes": [], "edges": []}
    view.set(None)
    assert widget.view == plot.view
    rc.close()
