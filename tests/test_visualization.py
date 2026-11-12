"""Regression tests for visualization inputs that used to be silently ignored."""

import html
import json
import re

import pytest

import edgecase as ec


def _abc_path():
    return ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])


def _drawing(player) -> str:
    """A step player's HTML without its per-call widget ID, so two calls can be compared."""
    return re.sub(r"\b(?:stv|tv|tc)\d+", "uid", player._repr_html_())


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

    def test_tooltip_shows_label(self):
        """Regression: the tooltip gave only the ID, so a node drawn with its label was hard to match up."""

        def tooltips(g):
            nodes = re.findall(r"data-node=\"([^\"]*)\" data-attrs='([^']*)'", g.plot(seed=1).svg)
            return {nid: json.loads(html.unescape(attrs)) for nid, attrs in nodes}

        assert tooltips(ec.karate_club())["17"] == {
            "id": "17",
            "label": "16",
            "faction": 1,
            "role": "Member",
        }
        # A label that just repeats the ID is left out
        assert "label" not in tooltips(ec.Graph().add_path_graph(n=3))["1"]

    def test_tooltip_numbers_are_rounded(self):
        """Regression: a stored position showed as `-0.3142857142857143,0.0571428571428571`."""
        g = ec.Graph().add_nodes(["A", "B", "C"])
        g.set_node_attrs("pos", {1: (-0.3142857142857143, 0.0571428571428571), 2: (0.5, 1e-17)})
        g.set_node_attrs("score", {1: 0.123456789, 2: 1234.56789, 3: 2})
        nodes = re.findall(r"data-node=\"([^\"]*)\" data-attrs='([^']*)'", g.plot(seed=1).svg)
        tips = {nid: json.loads(html.unescape(attrs)) for nid, attrs in nodes}
        # Coordinates share a decimal place (so one near 0 doesn't show as 1e-17)
        assert tips["1"]["pos"] == [-0.314, 0.057]
        assert tips["2"]["pos"] == [0.5, 0.0]
        # Other numbers keep 4 significant digits, but all of their whole-number digits
        assert [tips[n]["score"] for n in "123"] == [0.1235, 1235.0, 2]

    def test_tooltip_vector_precision_follows_point_spacing(self):
        from edgecase.svg_renderer import _tooltip_rows

        def first_x(n, spread):
            rows = [{"pos": (1 / 3 + spread * i / (n - 1), 0.0)} for i in range(n)]
            return _tooltip_rows(rows)[0]["pos"][0]

        # To a hundredth of the typical spacing: finer for more points or a smaller range
        assert first_x(4, 2.0) == 0.33
        assert first_x(1000, 2.0) == 0.3333
        assert first_x(30, 700.0) == 0

    @staticmethod
    def _edge_tooltips(svg: str) -> list[dict]:
        tips = re.findall(r" data-edge=\"[^\"]*\" data-attrs='([^']*)'", svg)
        return [json.loads(html.unescape(tip)) for tip in tips]

    def test_edge_tooltips(self):
        """Regression: an edge had no tooltip unless it had attributes, and was hard to hover,
        being as thin as it is drawn."""
        g = ec.Graph(directed=True).add_nodes(["Ana", "Ben", "Cy"])
        g.add_edges([("Ana", "Ben")], weight=0.123456789, rel="knows")
        g.add_edges([("Ben", "Cy")])
        svg = g.plot(label_by="label", seed=1).svg
        assert sorted(self._edge_tooltips(svg), key=lambda tip: tip["id"]) == [
            {"id": 1, "edge": "Ana → Ben", "weight": 0.1235, "rel": "knows"},
            {"id": 2, "edge": "Ben → Cy"},
        ]
        # On a transparent stroke wider than the edge drawn
        hit_areas = re.findall(r"<line[^>]* data-edge=[^>]*>", svg)
        assert len(hit_areas) == 2
        assert all('stroke="transparent" stroke-width="8.0"' in h for h in hit_areas)
        # The ends are named as their labels are drawn, here the IDs
        assert self._edge_tooltips(_abc_path().plot(seed=1).svg)[0]["edge"] == "1 — 2"

    @pytest.mark.parametrize("opts", [{"curved_edges": True}, {"edge_routing": "around"}])
    def test_edge_tooltips_on_curved_and_routed_edges(self, opts):
        g = ec.Graph(directed=True).add_nodes(list("ABCD")).add_edges([("A", "B"), ("B", "A")])
        g.add_edges([("C", "C"), ("A", "D")])
        assert len(self._edge_tooltips(g.plot(seed=1, **opts).svg)) == 4

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

    def test_layout_holds_up_on_a_page(self):
        """Regression: a page's larger text spread out the panel below the drawing, past a fixed
        height that cut off its last line, and the play and next buttons had the same arrow."""
        html = _abc_path().play()._repr_html_()
        root = re.match(r'<div id="\w+" style="([^"]*)"', html).group(1)
        assert "font-size:13px;line-height:1.4" in root
        assert "height:130px" not in html
        buttons = dict(re.findall(r'<button id="\w+-(prev|play|next)"[^>]*>(.*?)</button>', html))
        assert len(set(buttons.values())) == 3

    def test_code_keeps_the_theme_colors(self):
        """Regression: on a dark docs page, the page's `pre code` color reached the code shown
        for each step, so its unhighlighted parts were light gray on the light theme's panel."""
        html = _abc_path().play()._repr_html_()
        for selector in [".step-code-panel pre code", ".step-modal-body pre"]:
            rule = re.search(rf"#\w+ {re.escape(selector)} \{{([^}}]*)\}}", html).group(1)
            assert "color:inherit;" in rule and "background:none;" in rule

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

    @pytest.mark.parametrize("layout", ["force", "random"])
    def test_seed_makes_stable_layout_reproducible(self, layout):
        """Regression: play() had no seed=, so its stable layouts changed on every call."""
        g = ec.Graph().add_nodes(list("ABCDE")).add_edges([("A", "B"), ("B", "C"), ("C", "D")])
        one = _drawing(g.play(layout=layout, layout_mode="stable", seed=7))
        assert one == _drawing(g.play(layout=layout, layout_mode="stable", seed=7))
        assert one != _drawing(g.play(layout=layout, layout_mode="stable", seed=8))
        # Filtered replays take the seed too
        sub = g.steps.filter(kind="add")
        assert _drawing(sub.play(layout_mode="stable", seed=7)) == _drawing(
            sub.play(layout_mode="stable", seed=7)
        )

    def test_progressive_layout_reproducible_without_seed(self):
        g = ec.Graph().add_nodes(list("ABCDE")).add_edges([("A", "B"), ("B", "C"), ("C", "D")])
        assert _drawing(g.play()) == _drawing(g.play())
        assert _drawing(g.play(seed=7)) == _drawing(g.play(seed=7))
        assert _drawing(g.play(seed=7)) != _drawing(g.play(seed=8))


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
        edges = self._edges(out.read_text(encoding="utf-8"))
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
        html = g.plot_traversal(source="A")._repr_html_()
        steps = json.loads(re.search(r"var STEPS=(\[.*?\]);", html, re.S).group(1))
        a, b, c, d = (str(g.get_id_from_label(x)) for x in "ABCD")
        assert sorted(map(tuple, steps[-1]["tree_edges"])) == sorted([(a, b), (a, d), (b, c)])
        # Visited nodes are listed in visit order (not set order)
        assert steps[-1]["visited"] == [str(v) for v in g.bfs(int(a), return_list=True)]

    @pytest.mark.parametrize("layout", ["force", "random"])
    def test_seed_makes_layout_reproducible(self, layout):
        """Regression: plot_traversal() had no seed=, so its layout changed on every call."""
        g = ec.karate_club()
        one = _drawing(g.plot_traversal(source=1, layout=layout, seed=7))
        assert one == _drawing(g.plot_traversal(source=1, layout=layout, seed=7))
        assert one != _drawing(g.plot_traversal(source=1, layout=layout, seed=8))

    def test_recorder_seed_makes_layout_reproducible(self):
        g = ec.karate_club()
        rec = g.record_traversal()
        rec.select_nodes_by_id([1]).trav_out().trav_out_edge()
        assert _drawing(rec.plot(seed=7)) == _drawing(rec.plot(seed=7))
        assert _drawing(rec.plot(seed=7)) != _drawing(rec.plot(seed=8))

    @pytest.mark.parametrize("theme", ["light", "dark"])
    @pytest.mark.parametrize("player", ["plot_traversal", "recorder"])
    def test_controls_have_the_theme_background(self, player, theme):
        """Regression: the controls were drawn on the page's background, so on a dark page the
        light theme's dark text (and the dark theme's on a light page) was hard to read."""
        from edgecase.svg_renderer import THEMES

        g = ec.karate_club()
        if player == "plot_traversal":
            out = g.plot_traversal(source=1, theme=theme, seed=1)
        else:
            rec = g.record_traversal()
            rec.select_nodes_by_id([1]).trav_out()
            out = rec.plot(theme=theme, seed=1)
        outer = re.match(r'<div class="ec-plot" style="([^"]*)"', out._repr_html_()).group(1)
        assert f"background:{THEMES[theme].bg}" in outer
        assert f"color:{THEMES[theme].label_color}" in outer
        assert f"color-scheme:{theme}" in outer


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
            (lambda g: g.plot_traversal(source="A", theme="blue"), "Unknown theme"),
            (lambda g: g.plot_vis(layout="spring"), "Unknown layout"),
        ],
    )
    def test_raises(self, call, match):
        with pytest.raises(ValueError, match=match):
            call(_abc_path())

    def test_edge_color_by_accepts_builtins(self):
        assert "<svg" in _abc_path().plot(edge_color_by="degree", seed=1).svg
