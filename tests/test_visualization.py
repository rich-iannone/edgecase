"""Regression tests for visualization inputs that used to be silently ignored."""

import html
import json
import math
import random
import re

import pytest

import edgecase as ec


def _abc_path():
    return ec.Graph().add_nodes(["A", "B", "C"]).add_edges([("A", "B"), ("B", "C")])


def _drawing(player) -> str:
    """A step player's HTML (or some markup) without its per-call widget ID, so two calls can be
    compared."""
    markup = player if isinstance(player, str) else player._repr_html_()
    return re.sub(r"\b(?:stv|tv|tc)\d+", "uid", markup)


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
        from edgecase._core import tooltip_attrs

        def first_x(n, spread):
            rows = [{"pos": (1 / 3 + spread * i / (n - 1), 0.0)} for i in range(n)]
            return json.loads(html.unescape(tooltip_attrs(rows)[0]))["pos"][0]

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

    @staticmethod
    def _tip_table(page: str) -> dict:
        """{row number: tooltip} from the table of edge tooltips in a plot's HTML."""
        found = re.search(
            r'<script type="application/json" class="ec-edge-tips">(.*?)</script>', page, re.S
        )
        table = json.loads(found.group(1))
        rows = {}
        for key, column in zip(table["k"], table["c"]):
            cells = enumerate(column) if isinstance(column, list) else column.items()
            for i, value in cells:
                rows.setdefault(int(i), {})[key] = value
        return rows

    @staticmethod
    def _tooltip_graph():
        # Straight, routed and bowed (reciprocal) edges, a loop, and edges with and without
        # attributes
        g = ec.Graph(directed=True).add_nodes(["a", "b", "c", "d", "e"])
        g.add_edges([("a", "b")], weight=0.5, rel="knows")
        g.add_edges([("b", "a"), ("c", "c")])
        g.add_edges([("c", "d")], rel="</script><b>")
        g.add_edges([("d", "e"), ("a", "e"), ("b", "d")], since=2020)
        return g

    @staticmethod
    def _shapes(svg: str) -> list[float]:
        """Every number giving the shape of an edge drawn (not a hit area), sorted."""
        drawn = re.sub(r"<(?:line|path) [^>]*stroke=\"transparent\"[^>]*>", "", svg)
        values = re.findall(r' (?:d|points|x1|y1|x2|y2)="([^"]*)"', drawn)
        return sorted(float(v) for value in values for v in re.findall(r"-?\d+\.?\d*", value))

    @pytest.mark.parametrize("opts", [{}, {"curved_edges": True}, {"edge_width_by": "weight"}])
    def test_large_plots_are_lean(self, monkeypatch, opts):
        """Above `LEAN_MIN` nodes or edges, an edge has no hit area; it is hovered as drawn, and
        its tooltip is its row of one table. Edges and nodes sharing a style are gathered into a
        group carrying it. The shapes, the nodes and the tooltips are otherwise the same."""
        import edgecase.svg_renderer as svg_renderer

        g = self._tooltip_graph()
        full = g.plot(seed=1, label_by="label", **opts)
        monkeypatch.setattr(svg_renderer, "LEAN_MIN", 3)
        lean = g.plot(seed=1, label_by="label", **opts)

        assert 'stroke="transparent"' not in lean.svg
        assert "data-edge=" not in lean.svg
        assert 'class="ec-edge-tips"' not in full._repr_html_()
        # Each edge drawn is tagged with its row, its arrowhead too
        tagged = [int(i) for i in re.findall(r' data-e="(\d+)"', lean.svg)]
        assert sorted(set(tagged)) == list(range(7))
        assert len(tagged) == 14

        # The elements inherit their style from their groups
        elements = re.findall(r"<(?:path|circle) [^>]*>", lean.svg)
        assert not any(" stroke=" in e or " fill=" in e for e in elements)
        assert lean.svg.count('<g fill="none" stroke=') >= 1
        assert self._shapes(lean.svg) == self._shapes(full.svg)
        node = r'<circle cx="[^"]*" cy="[^"]*" r="[^"]*"'
        assert re.findall(node, lean.svg) == re.findall(node, full.svg)
        assert lean.svg.count("data-node=") == 5

        rows = self._tip_table(lean._repr_html_())
        assert sorted(rows) == list(range(7))
        tips = sorted(rows.values(), key=lambda tip: tip["id"])
        assert tips == sorted(self._edge_tooltips(full.svg), key=lambda tip: tip["id"])
        # A key that some rows lack is left out of theirs
        assert sum("since" in tip for tip in rows.values()) == 3

    def test_lean_curved_edges_keep_their_labels(self, monkeypatch):
        import edgecase.svg_renderer as svg_renderer

        monkeypatch.setattr(svg_renderer, "LEAN_MIN", 3)
        svg = self._tooltip_graph().plot(seed=1, curved_edges=True, edge_label_by="rel").svg
        for ref in re.findall(r'<textPath href="#([^"]+)"', svg):
            assert f'id="{ref}"' in svg

    def test_lean_plots_keep_their_order(self, monkeypatch):
        """A group holds a run of elements in a row, so highlighted edges stay over the rest."""
        import edgecase.svg_renderer as svg_renderer

        monkeypatch.setattr(svg_renderer, "LEAN_MIN", 3)
        g = ec.karate_club()
        svg = g.plot(seed=1, highlight=["0", "1", "2"]).svg
        groups = re.findall(
            r'<g fill="none" stroke="[^"]*" stroke-width="[^"]*" stroke-opacity="([^"]*)"', svg
        )
        opacities = [float(o) for o in groups]
        assert opacities == sorted(opacities)
        assert len(set(opacities)) > 1

    def test_tooltip_table_survives_any_value(self):
        from edgecase._core import tooltip_table

        rows = [{"id": 1, "rel": "</script>", "score": math.nan}, {"id": 2, "pos": [math.inf, 1]}]
        text = tooltip_table(rows)
        assert "<" not in text
        table = json.loads(text)
        assert table == {
            "k": ["id", "rel", "score", "pos"],
            "c": [[1, 2], {"0": "</script>"}, {"0": "nan"}, {"1": ["inf", 1]}],
        }

    def test_tooltip_json_is_what_python_writes(self):
        # The tooltips are written in Rust as json.dumps(..., default=str) wrote them in Python:
        # these are its output, from before the move
        from edgecase._core import tooltip_attrs, tooltip_table

        rows = [
            {"id": "1", "x": 0.123456789, "pos": (0.1, 1e-17), "s": "é</script>&'", "n": None},
            {"id": "2", "x": math.nan, "pos": [3, 4.56789], "inf": math.inf, "b": True},
            {"id": 3, "nested": {"a": [1.5, math.nan], 2: "two", None: 1, True: 0}, "set": {1}},
            {"big": 10**30},
        ]
        assert tooltip_attrs(rows) == [
            '{"id": "1", "x": 0.1235, "pos": [0.1, 0.0], "s": "\\u00e9&lt;/script>&amp;&#39;", '
            '"n": null}',
            '{"id": "2", "x": NaN, "pos": [3, 4.57], "inf": Infinity, "b": true}',
            '{"id": 3, "nested": {"a": [1.5, NaN], "2": "two", "null": 1, "true": 0}, '
            '"set": "{1}"}',
            '{"big": 1000000000000000000000000000000}',
        ]
        assert tooltip_table(rows) == (
            '{"k": ["id", "x", "pos", "s", "n", "inf", "b", "nested", "set", "big"], "c": '
            '[{"0": "1", "1": "2", "2": 3}, {"0": 0.1235, "1": "nan"}, '
            '{"0": [0.1, 0.0], "1": [3, 4.57]}, {"0": "\\u00e9\\u003c/script>&\'"}, '
            '{"0": null}, {"1": "inf"}, {"1": true}, '
            '{"2": {"a": [1.5, NaN], "2": "two", "null": 1, "true": 0}}, {"2": "{1}"}, '
            '{"3": 1000000000000000000000000000000}]}'
        )

    @pytest.mark.parametrize("opts", [{"curved_edges": True}, {"edge_routing": "around"}])
    def test_edge_tooltips_on_curved_and_routed_edges(self, opts):
        g = ec.Graph(directed=True).add_nodes(list("ABCD")).add_edges([("A", "B"), ("B", "A")])
        g.add_edges([("C", "C"), ("A", "D")])
        assert len(self._edge_tooltips(g.plot(seed=1, **opts).svg)) == 4

    def test_routed_edge_goes_around_a_node(self):
        # A is drawn between B and C, so the B-C edge bends around it unless asked not to
        g = ec.Graph().add_nodes(list("ABC"))
        g.add_edges([("B", "C"), ("A", "B")])
        g.set_node_attrs("pos", {"A": (0.0, 0.0), "B": (-1.0, 0.0), "C": (1.0, 0.0)})
        curve = re.compile(r'<path d="M[^"]* C')
        opts = {"layout": "pos", "node_overlap": "allow"}
        assert curve.search(g.plot(**opts).svg)
        assert not curve.search(g.plot(**opts, edge_routing="straight").svg)

    def test_auto_routing_keeps_to_a_budget(self, monkeypatch):
        """`edge_routing="auto"` routes edges while their waypoints fit the budget, and draws
        them straight past it. `"around"` routes them whatever it costs."""
        import edgecase.svg_renderer as svg_renderer

        g = ec.Graph().add_nodes(list("ABC"))
        g.add_edges([("B", "C"), ("A", "B")])
        g.set_node_attrs("pos", {"A": (0.0, 0.0), "B": (-1.0, 0.0), "C": (1.0, 0.0)})
        curve = re.compile(r'<path d="M[^"]* C')
        opts = {"layout": "pos", "node_overlap": "allow"}
        assert g.plot(**opts).svg == g.plot(**opts, edge_routing="around").svg
        assert curve.search(g.plot(**opts).svg)
        # The one route has three waypoints
        monkeypatch.setattr(svg_renderer, "ROUTE_WAYPOINTS_MAX", 2)
        assert not curve.search(g.plot(**opts).svg)
        assert g.plot(**opts).svg == g.plot(**opts, edge_routing="straight").svg
        assert curve.search(g.plot(**opts, edge_routing="around").svg)
        assert curve.search(g.plot(**opts, edge_routing=3).svg)

    def test_unknown_edge_routing_raises(self):
        with pytest.raises(ValueError, match="Unknown edge_routing 'none'"):
            _abc_path().plot(edge_routing="none")

    def test_route_edges_stops_past_its_budget(self):
        from edgecase._core import route_edges

        xs, ys, radii = [0.0, 100.0, 50.0, 50.0], [0.0, 0.0, 1.0, 80.0], [5.0] * 4
        edges = [(0, 1), (2, 3)]
        routes = route_edges(xs, ys, radii, edges, margin=4.0)
        assert [None if r is None else len(r) for r in routes] == [3, None]
        assert route_edges(xs, ys, radii, edges, margin=4.0, max_waypoints=3) == routes
        assert route_edges(xs, ys, radii, edges, margin=4.0, max_waypoints=2) is None

    def test_plot_refuses_what_would_be_too_large(self, monkeypatch):
        import edgecase.svg_renderer as svg_renderer

        monkeypatch.setattr(svg_renderer, "PLOT_BYTES_MAX", 1000)
        with pytest.raises(ValueError, match=r"at least 0 MB for 34 nodes and 78 edges.*k_core"):
            ec.karate_club().plot(backend="svg")
        _abc_path().plot()

    def test_plot_notes_output_past_the_notebook_budget(self, monkeypatch):
        import warnings

        import edgecase.svg_renderer as svg_renderer

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ec.karate_club().plot(seed=1)
        monkeypatch.setattr(svg_renderer, "NOTEBOOK_BYTES_MAX", 1000)
        with pytest.warns(UserWarning, match="past the 0 MB a notebook output") as caught:
            ec.karate_club().plot(seed=1)
        # Pointing at the call to plot()
        assert caught[0].filename == __file__

    def test_medium_graph_fits_the_notebook_budget(self):
        """A graph of thousands of nodes and edges is drawn whole, straight and lean."""
        player = ec.power_grid().plot(seed=1, show_labels=False)
        assert len(player._repr_html_()) < 2_000_000
        assert " C" not in player.svg
        assert 'stroke="transparent"' not in player.svg

    def test_route_edges_finds_every_node_in_the_way(self):
        """The grid that finds the nodes near an edge misses none that checking them all finds."""
        from edgecase._core import route_edges

        rng = random.Random(1)
        n = 400
        xs = [rng.uniform(0, 500) for _ in range(n)]
        ys = [rng.uniform(0, 300) for _ in range(n)]
        radii = [rng.uniform(1, 12) for _ in range(n)]
        edges = [(rng.randrange(n), rng.randrange(n)) for _ in range(600)]
        margin = 3.0

        def in_the_way(s, t):
            (ax, ay), (bx, by) = (xs[s], ys[s]), (xs[t], ys[t])
            dx, dy = bx - ax, by - ay
            found = []
            for i in range(n):
                if i in (s, t) or dx * dx + dy * dy < 1e-12:
                    continue
                u = ((xs[i] - ax) * dx + (ys[i] - ay) * dy) / (dx * dx + dy * dy)
                u = min(max(u, 0.0), 1.0)
                dist = math.dist((xs[i], ys[i]), (ax + u * dx, ay + u * dy))
                if 0 < u < 1 and dist < radii[i] + margin:
                    found.append(i)
            return found

        routes = route_edges(xs, ys, radii, edges, margin=margin)
        for (s, t), route in zip(edges, routes):
            expected = in_the_way(s, t)
            if not expected:
                assert route is None
            else:
                # The ends, and a waypoint for each node in the way
                assert len(route) == len(expected) + 2
                assert (route[0], route[-1]) == ((xs[s], ys[s]), (xs[t], ys[t]))

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
        root = re.search(r'<div id="\w+" style="([^"]*)"', html).group(1)
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

    def test_stored_layout(self):
        """play() draws nodes where a layout_*() method put them, the same in every frame."""
        g = _abc_path()
        g.set_node_attrs("xy", {"A": [0.0, 0.0], "B": [1.0, 0.0], "C": [2.0, 0.0]})
        html = g.play(layout="xy")._repr_html_()
        spots = re.findall(r'class="step-node" data-label="(\w)" cx="([\d.]+)" cy="([\d.]+)"', html)
        assert [(lbl, float(x), float(y)) for lbl, x, y in spots] == [
            ("A", 60.0, 250.0),
            ("B", 350.0, 250.0),
            ("C", 640.0, 250.0),
        ]
        assert "var LAYOUTS=[];" in html
        with pytest.raises(ValueError, match="Unknown layout 'nope'"):
            g.play(layout="nope")

    def test_edge_glows_cover_the_canvas(self):
        """Regression: a glow sized to a horizontal edge's bounding box (zero height) hid the
        edge, so selected edges along a row were not drawn at all."""
        html = _abc_path().play()._repr_html_()
        for name in ("add", "rm", "sel"):
            assert re.search(rf'<filter id="\w+-glow-{name}-e" filterUnits="userSpaceOnUse"', html)
        assert "setAttribute('filter',GLOW_SEL_E)" in html

    def test_clear_steps_keeps_existing_elements(self):
        """Regression: nodes and edges from before clear_steps() never appeared in play()."""
        g = _abc_path().clear_steps()
        assert g.steps.baseline == (["A", "B", "C"], [("A", "B"), ("B", "C")])
        g.add_node(label="D").add_edge("C", "D")
        html = g.play(layout="circular")._repr_html_()
        assert "No steps recorded" not in html
        for label in "ABCD":
            assert f'"{label}"' in html
        # They are on the first frame too, rather than an empty graph
        first = json.loads(re.search(r"var STEPS=(\[.*?\]);\n", html, re.S).group(1))[0]
        assert (first["nodes"], first["edges"]) == (["A", "B", "C"], [["A", "B"], ["B", "C"]])
        assert first["narration"] == (
            "The graph as it was when the step log was cleared: 3 nodes, 2 edges."
        )

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

    def test_traversal_note_leads_its_first_frame(self):
        """Regression: a note() on a traversal was dropped, since the player shows one frame per
        visited node, each with its own narration."""
        def visits(graph):
            html = graph.play()._repr_html_()
            steps = json.loads(re.search(r"var STEPS=(\[.*?\]);\n", html, re.S).group(1))
            return [s["narration"] for s in steps if s["kind"] == "traverse"]

        noted = _abc_path().note("Spread out from A.").bfs("A")
        assert visits(noted) == ["Spread out from A. Start at A", "Visit B from A", "Visit C from B"]
        # Without a note, the first frame keeps its own narration alone
        assert visits(_abc_path().bfs("A"))[0] == "Start at A"

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


def _renderings(theme, **opts):
    """Every renderer's output for `theme=`."""
    g = ec.karate_club().bfs(source=1)
    rec = g.record_traversal()
    rec.select_nodes_by_id([1]).trav_out()
    return {
        "plot": g.plot(theme=theme, seed=1, **opts),
        "play": g.play(theme=theme, seed=1, **opts),
        "plot_traversal": g.plot_traversal(source=1, theme=theme, seed=1, **opts),
        "recorder": rec.plot(theme=theme, seed=1, **opts),
    }


class TestAutoTheme:
    """theme="auto" (and the default, theme=None) gives one rendering for light and dark pages."""

    @pytest.mark.parametrize("theme", [None, "auto"])
    def test_colors_are_variables(self, theme):
        from edgecase.svg_renderer import DARK, LIGHT

        for name, out in _renderings(theme).items():
            for markup in (out.svg, out._repr_html_()):
                assert "var(--ect-" in markup, name
                # both palettes are defined, the dark one for dark systems and dark pages
                style = re.search(r'<style class="ec-theme-vars">(.*?)</style>', markup).group(1)
                assert LIGHT.bg in style and DARK.bg in style
                assert "@media (prefers-color-scheme: dark)" in style
                assert "body.quarto-dark" in style and '[data-jp-theme-light="false"]' in style
                assert f'fill="{LIGHT.bg}"' not in markup and f'fill="{DARK.bg}"' not in markup
            assert out.svg.startswith('<svg class="ec-theme"'), name
            assert out._repr_html_().startswith('<div class="ec-theme"'), name

    @pytest.mark.parametrize("theme", ["light", "dark"])
    def test_fixed_themes_use_plain_colors(self, theme):
        for name, out in _renderings(theme).items():
            assert "var(--" not in out._repr_html_() and "ec-theme" not in out._repr_html_(), name

    def test_default_saves_the_light_theme(self, tmp_path):
        """Other programs than browsers don't apply CSS variables, so files get plain colors."""
        default, light = _renderings(None), _renderings("light")
        for name, out in default.items():
            out.save(tmp_path / f"{name}.svg")
            out.save(tmp_path / f"{name}.html")
            saved = (tmp_path / f"{name}.svg").read_text()
            assert _drawing(saved) == _drawing(light[name].svg), name
            assert "var(--" not in (tmp_path / f"{name}.html").read_text(), name
        ec.karate_club().plot(seed=1, output=str(tmp_path / "out.svg"))
        light_plot = ec.karate_club().plot(theme="light", seed=1)
        assert (tmp_path / "out.svg").read_text() == light_plot.svg

    def test_explicit_auto_saves_auto(self, tmp_path):
        out = ec.karate_club().plot(theme="auto", seed=1, output=str(tmp_path / "out.svg"))
        assert (tmp_path / "out.svg").read_text() == out.svg

    def test_same_drawing_same_markup(self):
        """The variables are named after their colors, so seeded drawings stay reproducible and
        renderings on one page don't redefine each other's variables."""

        def variables(markup):
            style = re.search(r'<style class="ec-theme-vars">(.*?)</style>', markup).group(1)
            light, dark = (
                dict(re.findall(r"(--ect-\w+):([^;}]+)", block))
                for block in re.findall(r"\.ec-theme\)?\{([^}]*)\}", style)[:2]
            )
            return {name: (light[name], dark[name]) for name in light}

        g = ec.karate_club()
        assert g.plot(seed=1).svg == g.plot(seed=1).svg
        plot, play = variables(g.plot(seed=1).svg), variables(g.play(seed=1)._repr_html_())
        shared = plot.keys() & play.keys()
        assert shared and all(plot[name] == play[name] for name in shared)

    def test_page_scheme_outranks_the_system(self):
        """The system's scheme applies only where the page doesn't set one, whatever the order of
        the renderings: the base rule has no specificity, the page markers the most."""
        svg = ec.karate_club().plot(seed=1).svg
        style = re.search(r'<style class="ec-theme-vars">(.*?)</style>', svg).group(1)
        order = [":where(.ec-theme){", "@media (prefers-color-scheme: dark){.ec-theme{"]
        order += [":is(html.quarto-light", ":is(html.quarto-dark"]
        positions = [style.index(rule) for rule in order]
        assert positions == sorted(positions)

    def test_unknown_theme_lists_auto(self):
        with pytest.raises(ValueError, match="use 'auto', 'light' or 'dark'"):
            _abc_path().plot(theme="blue")


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


def _node_fills(svg: str) -> dict:
    """{node ID: fill} of a plot's node circles."""
    return {
        int(nid): fill
        for fill, nid in re.findall(r'<circle [^>]*?fill="([^"]+)"[^>]*?data-node="(\d+)"', svg)
    }


def _edge_strokes(svg: str) -> dict:
    """{edge ID: (stroke, stroke width)} of a plot's edges, read off each drawn edge and the
    hover target that follows it."""
    found = re.findall(
        r'<(?:line|path) [^>]*?stroke="([^"]+)" stroke-width="([\d.]+)"[^>]*?/>'
        r"(?:<polygon [^>]*/></g>)?\s*"
        r'<(?:line|path) [^>]*?stroke="transparent"[^>]*?data-attrs=\'(\{[^\']*\})\'',
        svg,
    )
    return {json.loads(attrs)["id"]: (stroke, float(w)) for stroke, w, attrs in found}


def _chain():
    """a - b - c - d - e, with a branch c - x."""
    return ec.Graph.from_edges([("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("c", "x")])


def _ids(graph) -> dict:
    """{label: node ID}."""
    return {row["label"]: row["id"] for row in graph.get_node_df().iter_rows(named=True)}


class TestHighlight:
    """highlight= and highlight_edges= pick out nodes and edges in the accent color and mute
    the rest."""

    ACCENT = "#c2410c"
    MUTED = "#d4d4d8"

    def _plot(self, graph, **opts):
        return graph.plot(theme="light", seed=1, legend=False, **opts).svg

    def _picked(self, graph, **opts):
        from edgecase.svg_renderer import _resolve_highlight

        picked = _resolve_highlight(graph, opts.get("highlight"), opts.get("highlight_edges"))
        label = {nid: graph.get_node(nid).label for nid in graph.get_node_ids()}
        ends = {
            eid: tuple(label[n] for n in graph.get_node_ids_by_edge_id(eid))
            for eid in graph.get_edge_ids()
        }
        return {
            level: ({label[n] for n in nodes}, {ends[e] for e in edges})
            for level, (nodes, edges) in picked.items()
        }

    def test_path_of_nodes_takes_its_edges(self):
        picked = self._picked(_chain(), highlight=["a", "b", "c"])
        assert picked["accent"] == ({"a", "b", "c"}, {("a", "b"), ("b", "c")})
        assert picked["soft"] == (set(), set())

    def test_nodes_not_in_a_row_take_no_edges(self):
        nodes, edges = self._picked(_chain(), highlight=["a", "c", "e"])["accent"]
        assert nodes == {"a", "c", "e"} and edges == set()

    def test_one_node_by_id_or_label(self):
        g = _chain()
        by_label = self._picked(g, highlight="d")["accent"]
        by_id = self._picked(g, highlight=_ids(g)["d"])["accent"]
        assert by_label == by_id == ({"d"}, set())

    def test_pairs_pick_out_edges_and_their_ends(self):
        nodes, edges = self._picked(_chain(), highlight=[("c", "x"), ("a", "b")])["accent"]
        assert nodes == {"a", "b", "c", "x"} and edges == {("c", "x"), ("a", "b")}

    def test_path_order_attribute(self):
        """An attribute of distinct whole numbers, like shortest_path_order, is a path."""
        g = _chain().shortest_path("a", "e")
        nodes, edges = self._picked(g, highlight="shortest_path_order")["accent"]
        assert nodes == {"a", "b", "c", "d", "e"}
        assert edges == {("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")}

    def test_marked_attribute_values(self):
        g = _chain()
        g.set_node_attrs("flag", dict(zip([_ids(g)[n] for n in "abcd"], [0, None, False, 1])))
        nodes, edges = self._picked(g, highlight="flag")["accent"]
        # 0 counts, while None and False don't; two values not adjacent make no path
        assert nodes == {"a", "d"} and edges == set()

    def test_edge_attribute_and_selection(self):
        g = _chain()
        ids = _ids(g)
        g.set_edge_attrs("spur", {g.find_edge_ids(ids["c"], ids["x"])[0]: True})
        assert self._picked(g, highlight="spur")["accent"] == ({"c", "x"}, {("c", "x")})
        assert self._picked(g, highlight_edges="spur")["accent"] == ({"c", "x"}, {("c", "x")})
        g.select_nodes_by_id([ids["a"], ids["e"]])
        assert self._picked(g, highlight="selection")["accent"][0] == {"a", "e"}

    def test_edge_ids_and_dicts(self):
        g = _chain()
        first, *_, last = sorted(g.get_edge_ids())
        for spec in (first, [first], {first: 3.5}):
            assert self._picked(g, highlight_edges=spec)["accent"] == ({"a", "b"}, {("a", "b")})
        both = self._picked(g, highlight_edges=[("a", "b"), last])["accent"][1]
        assert both == {("a", "b"), ("c", "x")}

    def test_levels(self):
        picked = self._picked(
            _chain(),
            highlight={"accent": ["a", "b"], "soft": ["b", "c", "d"]},
        )
        # what both levels pick out is accent
        assert picked["accent"] == ({"a", "b"}, {("a", "b")})
        assert picked["soft"] == ({"c", "d"}, {("b", "c"), ("c", "d")})

    @pytest.mark.parametrize(
        "opts",
        [
            {"highlight": "nobody"},
            {"highlight": ["a", "nobody"]},
            {"highlight": [("a", "e")]},
            {"highlight_edges": "no_such_attr"},
        ],
    )
    def test_unknown_things_raise(self, opts):
        with pytest.raises(ValueError):
            self._plot(_chain(), **opts)

    def test_drawing(self):
        g = _chain()
        svg = self._plot(g, highlight=["a", "b", "c"])
        fills = _node_fills(svg)
        ids = _ids(g)
        assert {fills[ids[n]] for n in "abc"} == {self.ACCENT}
        assert {fills[ids[n]] for n in "dex"} == {self.MUTED}
        strokes = _edge_strokes(svg)
        ab = g.find_edge_ids(ids["a"], ids["b"])[0]
        de = g.find_edge_ids(ids["d"], ids["e"])[0]
        assert strokes[ab][0] == self.ACCENT and strokes[de][0] != self.ACCENT
        assert strokes[ab][1] > strokes[de][1]

    def test_accent_nodes_keep_their_group_color(self):
        g = _chain()
        ids = _ids(g)
        g.set_node_attrs(
            "team", {nid: "red" if lbl in "abc" else "blue" for lbl, nid in ids.items()}
        )
        plain = _node_fills(self._plot(g, color_by="team"))
        fills = _node_fills(self._plot(g, color_by="team", highlight=["a", "b"]))
        assert fills[ids["a"]] == plain[ids["a"]] != self.ACCENT
        assert fills[ids["c"]] == self.MUTED

    def test_soft_nodes_are_pale_and_ringed(self):
        g = _chain()
        ids = _ids(g)
        svg = self._plot(g, highlight={"soft": "e"})
        circle = re.search(rf'<circle [^>]*data-node="{ids["e"]}"[^>]*>', svg).group(0)
        assert f'stroke="{self.ACCENT}"' in circle
        assert 'fill="#c2410c"' not in circle and f'fill="{self.MUTED}"' not in circle

    def test_play_final_frame(self):
        g = _chain().bfs("a")
        html = g.play(highlight=["a", "b"], highlight_edges={"soft": [("c", "x")]})._repr_html_()
        steps = json.loads(re.search(r"var STEPS=(\[.*?\]);\n", html, re.S).group(1))
        assert all("hl" not in s for s in steps[:-1])
        assert steps[-1]["hl"] == {
            "an": ["a", "b"],
            "ae": [["a", "b"]],
            "sn": ["c", "x"],
            "se": [["c", "x"]],
        }
        assert "var COL_ACC='#c2410c'" not in html  # auto theme: a variable
        assert "COL_ACC=" in html

    def test_play_without_highlight_is_unchanged(self):
        g = _chain()
        assert _drawing(g.play(seed=1)) == _drawing(g.play(seed=1, highlight=None))


class TestColorScales:
    """Numeric color_by= gets a color scale, and the legend explains the colors."""

    def _graph(self, values):
        g = _chain()
        g.set_node_attrs("score", dict(zip(sorted(g.get_node_ids()), values)))
        return g

    def test_floats_get_a_sequential_scale(self):
        from edgecase.svg_renderer import SEQUENTIAL_STOPS

        g = self._graph([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        fills = _node_fills(g.plot(color_by="score", theme="light", seed=1).svg)
        ids = sorted(g.get_node_ids())
        assert fills[ids[0]] == SEQUENTIAL_STOPS[0][1]
        assert fills[ids[-1]] == SEQUENTIAL_STOPS[-1][1]
        assert len(set(fills.values())) == 6

    def test_dark_theme_runs_the_other_way(self):
        from edgecase.svg_renderer import SEQUENTIAL_STOPS

        g = self._graph([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        fills = _node_fills(g.plot(color_by="score", theme="dark", seed=1).svg)
        assert fills[sorted(g.get_node_ids())[-1]] == SEQUENTIAL_STOPS[-1][2]

    def test_whole_numbers_are_groups_unless_asked(self):
        g = self._graph([3, 1, 2, 1, 3, 2])
        svg = g.plot(color_by="score", theme="light", seed=1).svg
        assert "linearGradient" not in svg and len(set(_node_fills(svg).values())) == 3
        svg = g.plot(color_by="score", color_scale="sequential", theme="light", seed=1).svg
        assert "linearGradient" in svg

    def test_diverging_is_symmetric_about_the_midpoint(self):
        from edgecase.svg_renderer import DIVERGING_STOPS

        g = self._graph([-1.0, -0.5, 0.0, 0.5, 2.0, 0.25])
        ids = sorted(g.get_node_ids())
        fills = _node_fills(
            g.plot(color_by="score", color_scale="diverging", theme="light", seed=1).svg
        )
        assert fills[ids[2]] == DIVERGING_STOPS[1][1]  # 0 is the neutral middle
        assert fills[ids[4]] == DIVERGING_STOPS[2][1]  # the more extreme end is full color
        assert fills[ids[0]] != DIVERGING_STOPS[0][1]
        fills = _node_fills(
            g.plot(color_by="score", color_scale="diverging", color_midpoint=0.5, theme="light").svg
        )
        assert fills[ids[3]] == DIVERGING_STOPS[1][1]

    def test_missing_values_are_gray(self):
        from edgecase.svg_renderer import MISSING_COLOR

        g = self._graph([0.1, None, 0.3, 0.4, 0.5, 0.6])
        svg = g.plot(color_by="score", theme="light", seed=1).svg
        assert _node_fills(svg)[sorted(g.get_node_ids())[1]] == MISSING_COLOR[0]
        assert "no value" in svg

    def test_missing_groups_are_gray(self):
        """Regression: nodes without a group took a palette hue, and a "none" legend entry."""
        from edgecase.svg_renderer import MISSING_COLOR

        g = self._graph(["a", None, "b", None, "a", "b"])
        svg = g.plot(color_by="score", theme="light", seed=1).svg
        fills = _node_fills(svg)
        ids = sorted(g.get_node_ids())
        assert fills[ids[1]] == fills[ids[3]] == MISSING_COLOR[0]
        assert len({fills[ids[0]], fills[ids[2]], MISSING_COLOR[0]}) == 3
        assert "no value" in svg and ">none</text>" not in svg

    def test_bad_scales_raise(self):
        g = self._graph([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        with pytest.raises(ValueError, match="Unknown color_scale"):
            g.plot(color_scale="rainbow")
        g.set_node_attrs("kind", {nid: "x" for nid in sorted(g.get_node_ids())})
        with pytest.raises(ValueError, match="needs numeric values"):
            g.plot(color_by="kind", color_scale="sequential")

    def test_groups_take_colors_in_sorted_order(self):
        """The same values get the same colors however the nodes are ordered."""
        a = self._graph(["pear", "apple", "fig", "apple", "pear", "fig"])
        b = self._graph(["fig", "pear", "apple", "fig", "apple", "pear"])
        color = {}
        for g in (a, b):
            fills = _node_fills(g.plot(color_by="score", theme="light", seed=1).svg)
            for nid, fill in fills.items():
                value = g.get_node(nid).attributes["score"]
                assert color.setdefault(value, fill) == fill


class TestLegend:
    def test_swatches_for_groups(self):
        svg = ec.karate_club().plot(color_by="faction", seed=1).svg
        legend = svg[svg.index('<g class="ec-legend"') :]
        assert ">faction</text>" in legend

    def test_ramp_for_scales(self):
        svg = ec.karate_club().pagerank().plot(color_by="pagerank", seed=1).svg
        assert 'class="ec-legend"' in svg and "linearGradient" in svg

    def test_legend_false_and_no_colors(self):
        g = ec.karate_club()
        assert 'class="ec-legend"' not in g.plot(color_by="faction", legend=False, seed=1).svg
        assert 'class="ec-legend"' not in g.plot(seed=1).svg

    def test_left_out_when_colors_repeat(self):
        """A legend can't tell apart more groups than the palette has colors."""
        svg = ec.karate_club().plot(color_by="label", seed=1).svg
        assert 'class="ec-legend"' not in svg

    def test_widens_the_canvas_not_the_drawing(self):
        g = ec.karate_club()
        plain = g.plot(color_by="faction", legend=False, seed=1, theme="light").svg
        with_legend = g.plot(color_by="faction", seed=1, theme="light").svg

        def view_box(svg):
            return [float(v) for v in re.search(r'viewBox="([^"]+)"', svg).group(1).split()]

        assert view_box(with_legend)[2] > view_box(plain)[2]

    def test_gradient_stops_follow_the_auto_theme(self):
        svg = ec.karate_club().pagerank().plot(color_by="pagerank", seed=1).svg
        stops = re.findall(r'<stop [^>]*stop-color="([^"]+)"', svg)
        assert stops and all(s.startswith("var(--ect-") for s in stops)


class TestEdgeAesthetics:
    def test_edge_color_by_edge_attribute(self):
        g = ec.pipeline()
        strokes = _edge_strokes(g.plot(edge_color_by="cost", theme="light", seed=1).svg)
        cost = {row["id"]: row["cost"] for row in g.get_edge_df().iter_rows(named=True)}
        by_value = {}
        for eid, (stroke, _) in strokes.items():
            assert by_value.setdefault(cost[eid], stroke) == stroke
        assert len(set(by_value.values())) == len(by_value) > 1

    def test_numeric_edge_attribute_gets_a_scale(self):
        g = ec.pipeline()
        g.set_edge_attrs("load", {eid: eid / 10 for eid in g.get_edge_ids()})
        svg = g.plot(edge_color_by="load", theme="light", seed=1).svg
        assert "linearGradient" in svg and '<g opacity="0.85">' in svg

    def test_edge_width_by(self):
        g = ec.pipeline()
        strokes = _edge_strokes(g.plot(edge_width_by="capacity", theme="light", seed=1).svg)
        cap = {row["id"]: row["capacity"] for row in g.get_edge_df().iter_rows(named=True)}
        widest = max(strokes, key=lambda e: strokes[e][1])
        narrowest = min(strokes, key=lambda e: strokes[e][1])
        assert cap[widest] == max(cap.values()) and cap[narrowest] == min(cap.values())
        assert strokes[widest][1] > strokes[narrowest][1]

    def test_cross_group_edges_default_to_gray(self):
        """Regression: edges between groups were salmon, close to the palette's red group."""
        g = ec.karate_club()
        for theme, gray in [("light", "#52525b"), ("dark", "#a1a1aa")]:
            svg = g.plot(color_by="faction", edge_color_by="faction", theme=theme, seed=1).svg
            strokes = {stroke for stroke, _ in _edge_strokes(svg).values()}
            assert gray in strokes and len(strokes) == 3

    def test_edge_label_by_weight(self):
        """Regression: `edge_label_by="weight"` drew no labels, since the weight isn't an
        attribute."""
        g = ec.Graph.from_edges([("A", "B", 2.5), ("B", "C", 7.25)])
        svg = g.plot(edge_label_by="weight", theme="light", seed=1).svg
        assert ">2.5</text>" in svg and ">7.25</text>" in svg

    def test_parallel_edge_labels_apart(self):
        """Regression: the labels of parallel edges were drawn on top of each other, at the
        middle of the straight line between the nodes instead of by each curve."""
        g = ec.Graph.from_edges([("A", "B", 3.0), ("A", "B", 4.0)])
        for curved in (False, True):
            svg = g.plot(edge_label_by="weight", curved_edges=curved, seed=1).svg
            if curved:
                paths = re.findall(r'<textPath href="#([^"]+)"', svg)
                ds = {
                    re.search(rf'id="{re.escape(p)}"[^>]*\bd="([^"]+)"', svg).group(1)
                    for p in paths
                }
                assert len(ds) == 2
            else:
                at = re.findall(r'<text x="([\d.-]+)" y="([\d.-]+)"[^>]*>[34]\.0</text>', svg)
                (xa, ya), (xb, yb) = [(float(x), float(y)) for x, y in at]
                assert math.hypot(xa - xb, ya - yb) > 20

    @pytest.mark.parametrize("end", [(4, 0), (0, 4), (3, 3)])
    def test_many_parallel_edge_labels_apart(self, end):
        """Regression: with three or more edges between two nodes, the labels of the inner
        edges were drawn on top of each other and of the outer ones."""
        g = ec.Graph()
        # A third node far away keeps the edges short, as in a larger drawing
        g.add_node(label="A", pos=[0, 0]).add_node(label="B", pos=list(end))
        g.add_node(label="C", pos=[12, 12])
        for w in range(5):
            g.add_edge("A", "B", amount=100 + w)
        svg = g.plot(layout="pos", edge_label_by="amount", seed=1).svg
        at = re.findall(r'<text x="([\d.-]+)" y="([\d.-]+)"[^>]*>10\d</text>', svg)
        spots = [(float(x), float(y)) for x, y in at]
        assert len(spots) == 5
        for i, (xa, ya) in enumerate(spots):
            for xb, yb in spots[i + 1 :]:
                assert abs(xa - xb) > 18 or abs(ya - yb) > 10

    def test_circular_layout_follows_id_order(self):
        """Regression: layout="circular" placed the nodes in storage order, so a cycle graph
        was drawn with its edges crossing the circle instead of as a ring."""
        svg = ec.Graph().add_cycle_graph(12).plot(layout="circular").svg
        spots = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)"[^>]*data-node="(\d+)"', svg)
        pos = {int(n): (float(x), float(y)) for x, y, n in spots}
        steps = [math.dist(pos[i], pos[i % 12 + 1]) for i in range(1, 13)]
        assert max(steps) - min(steps) < 1

    def test_circular_layout_follows_the_edges(self):
        """Regression: in ID order, a tree numbered breadth-first had its edges crossing the
        circle. The depth-first order of `layout_circular()` keeps parents next to children."""
        svg = ec.Graph().add_balanced_tree(2, 3).plot(layout="circular").svg
        spots = re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)"[^>]*data-node="(\d+)"', svg)
        pos = {int(n): (float(x), float(y)) for x, y, n in spots}
        cx = sum(x for x, _ in pos.values()) / len(pos)
        cy = sum(y for _, y in pos.values()) / len(pos)
        around = sorted(pos, key=lambda n: math.atan2(pos[n][0] - cx, cy - pos[n][1]) % math.tau)
        start = around.index(1)
        assert around[start:] + around[:start] == [1, 2, 4, 8, 9, 5, 10, 11, 3, 6, 12, 13, 7, 14, 15]

    def test_edge_width_by_unknown_raises(self):
        with pytest.raises(ValueError, match="no edge has an attribute"):
            ec.pipeline().plot(edge_width_by="nope")
