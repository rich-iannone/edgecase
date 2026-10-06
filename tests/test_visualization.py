"""Regression tests for visualization inputs that used to be silently ignored."""

import json
import re

import pytest

import edgecase as ec


def _abc_path():
    return ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])


class TestColorizeNodeAttrs:
    def test_alpha_appends_channel(self):
        """Regression: alpha= was accepted but ignored."""
        g = _abc_path()
        g.colorize_node_attrs(values={"A": "x", "C": "y"}, palette=["#ff0000", "#00F"], alpha=50)
        colors = g.get_node_attrs("color")
        assert colors[g.get_id_from_label("A")] == "#ff000080"
        assert colors[g.get_id_from_label("C")] == "#0000ff80"

    def test_alpha_validation(self):
        g = _abc_path()
        with pytest.raises(ValueError, match="between 0 and 100"):
            g.colorize_node_attrs(values={"A": "x"}, alpha=150)
        with pytest.raises(ValueError, match="hex colors"):
            g.colorize_node_attrs(values={"A": "x"}, palette=["red"], alpha=50)

    def test_unknown_palette_raises(self):
        """Regression: an unknown palette name silently fell back to "default"."""
        with pytest.raises(ValueError, match="Unknown palette 'nope'"):
            _abc_path().colorize_node_attrs(values={"A": "x"}, palette="nope")


def _labels_in_svg(svg: str) -> list[str]:
    return re.findall(r"<text[^>]*>([^<]*)</text>", svg)


class TestPlotNodeValues:
    def test_label_by_label(self):
        """Regression: label_by="label" showed IDs, since labels aren't stored attributes."""
        svg = _abc_path().plot(label_by="label", seed=1).svg
        assert {"A", "B", "C"} <= set(_labels_in_svg(svg))

    def test_display_attr_is_default_label(self):
        """Regression: set_node_attr_to_display() stored a preference nothing read."""
        g = _abc_path()
        g.set_node_attrs("name", {g.get_id_from_label(x): x.lower() for x in "ABC"})
        g.set_node_attr_to_display("name")
        assert {"a", "b", "c"} <= set(_labels_in_svg(g.plot(seed=1).svg))
        # An explicit label_by= still wins
        assert {"1", "2", "3"} <= set(_labels_in_svg(g.plot(label_by="id", seed=1).svg))

    def test_size_by_degree(self):
        """Regression: size_by="degree" did nothing."""
        svg = ec.Graph().add_star_graph(5).plot(size_by="degree", seed=1).svg
        radii = {float(r) for r in re.findall(r'<circle[^>]* r="([\d.]+)"', svg)}
        assert len(radii) == 2

    def test_unknown_attribute_raises(self):
        with pytest.raises(ValueError, match="no node has an attribute 'nope'"):
            _abc_path().plot(color_by="nope")
        with pytest.raises(ValueError, match="size_by="):
            _abc_path().plot(size_by="nope")

    def test_unknown_palette_raises(self):
        g = _abc_path().cmty_louvain()
        with pytest.raises(ValueError, match="Unknown palette"):
            g.plot(color_by="cmty_louvain", palette="nope")

    @pytest.mark.parametrize("layout", ["force", "random"])
    def test_seed_makes_layout_reproducible(self, layout):
        """Regression: the force and random layouts weren't seeded."""
        g = ec.karate_club()
        assert g.plot(layout=layout, seed=7).svg == g.plot(layout=layout, seed=7).svg
        assert g.plot(layout=layout, seed=7).svg != g.plot(layout=layout, seed=8).svg


class TestPlay:
    def test_custom_speed_is_selected(self):
        """Regression: any speed other than 0.5/1/2/4 fell back to 0.5x."""
        html = _abc_path().play(speed=1.5)._repr_html_()
        assert '<option value="800" selected>1.5&times;</option>' in html
        assert html.count(" selected>") == 1

    def test_invalid_speed_raises(self):
        with pytest.raises(ValueError, match="speed must be a positive number"):
            _abc_path().play(speed=0)

    def test_clear_steps_keeps_existing_elements(self):
        """Regression: nodes and edges from before clear_steps() never appeared in play()."""
        g = _abc_path().clear_steps()
        assert g.steps.baseline == (["A", "B", "C"], [("A", "B"), ("B", "C")])
        g.add_node(label="D").add_edge("C", "D")
        html = g.play(layout="circular")._repr_html_()
        assert "No steps recorded" not in html
        for label in "ABCD":
            assert f'"{label}"' in html


class TestPlotVis:
    def _edges(self, html: str) -> list[dict]:
        return json.loads(re.search(r"var edgesData = (\[.*?\]);", html, re.S).group(1))

    def test_every_edge_is_drawn(self, tmp_path):
        """Regression: parallel edges (and A->B plus B->A) were merged into one line."""
        from edgecase.viz import plot_graph

        g = ec.Graph(directed=True).add_nodes(["A", "B"])
        g.add_edges([("A", "B"), ("A", "B"), ("B", "A")])
        out = tmp_path / "vis.html"
        plot_graph(g, output_path=str(out), open_browser=False, display_inline=False)
        edges = self._edges(out.read_text())
        assert len(edges) == 3
        assert len({e["id"] for e in edges}) == 3
        assert sum("smooth" in e for e in edges) == 2

    def test_browser_target_skips_inline_display(self, monkeypatch, tmp_path):
        """Regression: inside IPython, target="browser" still displayed inline."""
        ipython = pytest.importorskip("IPython")
        import IPython.display

        from edgecase import viz

        displayed, opened = [], []
        monkeypatch.setattr(ipython, "get_ipython", lambda: object())
        monkeypatch.setattr(IPython.display, "display", lambda obj: displayed.append(obj))
        monkeypatch.setattr(viz.webbrowser, "open", lambda url: opened.append(url))
        _abc_path().plot_vis(target="browser")
        assert opened and not displayed
        _abc_path().plot_vis(target="notebook")
        assert len(displayed) == 1 and len(opened) == 1


class TestPlotTraversal:
    def test_directed_tree_edges_inferred(self):
        """Regression: on directed graphs the inferred tree edges were always empty."""
        g = ec.Graph(directed=True).add_nodes(["A", "B", "C", "D"])
        g.add_edges([("A", "B"), ("B", "C"), ("A", "D")])
        html = g.plot_traversal(start="A")._repr_html_()
        steps = json.loads(re.search(r"var STEPS=(\[.*?\]);", html, re.S).group(1))
        a, b, c, d = (str(g.get_id_from_label(x)) for x in "ABCD")
        assert sorted(map(tuple, steps[-1]["tree_edges"])) == sorted([(a, b), (a, d), (b, c)])


class TestUnknownStyleOptionsRaise:
    """Regression: unknown layout=/theme=/layout_mode= fell back silently, and edge_color_by=
    was ignored when no node had the attribute."""

    @pytest.mark.parametrize(
        "call, match",
        [
            (lambda g: g.plot(layout="spring"), "Unknown layout 'spring'"),
            (lambda g: g.plot(theme="blue"), "Unknown theme 'blue'"),
            (lambda g: g.plot(edge_color_by="nope"), "edge_color_by='nope'"),
            (lambda g: g.play(layout="spring"), "Unknown layout"),
            (lambda g: g.play(layout_mode="smooth"), "Unknown layout_mode 'smooth'"),
            (lambda g: g.plot_traversal(start="A", theme="blue"), "Unknown theme"),
            (lambda g: g.plot_vis(layout="spring"), "Unknown layout"),
        ],
    )
    def test_raises(self, call, match):
        with pytest.raises(ValueError, match=match):
            call(_abc_path())

    def test_edge_color_by_accepts_builtins(self):
        assert "<svg" in _abc_path().plot(edge_color_by="degree", seed=1).svg
