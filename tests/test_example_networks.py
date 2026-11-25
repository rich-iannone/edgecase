"""
The example networks, and the features planted in them.

The docs rely on these features (each loader's "What to Look For" list, and the examples that use
the networks), so a change that breaks one should fail here first.
"""

import itertools
import math

import edgecase as ec
import networkx as nx
import polars as pl
import pytest
from edgecase.example_networks import (
    _CAUSES,
    _METRO_SEGMENTS,
    _METRO_STATIONS,
    _ROADS,
    _TASKS,
    _TOWNS,
)

from .test_planarity import assert_crossing_free


def labels(graph, ids):
    return sorted(graph.get_label_from_id(i) for i in ids)


def edge_ends(graph, edge_ids):
    """The edges as sorted pairs of endpoint labels."""
    out = set()
    for eid in edge_ids:
        source, target = graph.get_node_ids_by_edge_id(eid)
        out.add(tuple(sorted((graph.get_label_from_id(source), graph.get_label_from_id(target)))))
    return out


def top(scores, graph):
    return graph.get_label_from_id(max(scores, key=scores.get))


@pytest.mark.parametrize(
    "loader, nodes, edges, directed, weighted",
    [
        (ec.metro, 20, 25, False, True),
        (ec.payments, 18, 35, True, True),
        (ec.courses, 17, 19, True, False),
        (ec.pipeline, 11, 14, True, False),
        (ec.exams, 14, 29, False, False),
        (ec.teams, 16, 21, False, True),
        (ec.friends, 24, 43, False, False),
        (ec.citations, 22, 58, True, False),
        (ec.currency, 6, 18, True, True),
        (ec.towns, 12, 26, False, True),
        (ec.offices, 20, 23, False, False),
        (ec.project, 16, 20, True, True),
        (ec.causal, 8, 10, True, False),
        (ec.konigsberg, 4, 7, False, False),
        (ec.org_chart, 20, 19, True, False),
        (ec.proteins, 31, 56, False, False),
        (ec.food_web, 16, 21, True, False),
        (ec.control_flow, 11, 14, True, False),
    ],
)
def test_shape(loader, nodes, edges, directed, weighted):
    graph = loader()
    assert graph.count_nodes() == nodes
    assert graph.count_edges() == edges
    assert graph.is_directed() == directed
    assert graph.is_weighted() == weighted
    # every node is labeled with a unique, readable name
    node_labels = graph.get_node_df()["label"].to_list()
    assert len(set(node_labels)) == nodes
    assert all(label and not label.isdigit() for label in node_labels)


@pytest.mark.parametrize(
    "loader",
    [
        ec.metro,
        ec.payments,
        ec.courses,
        ec.pipeline,
        ec.exams,
        ec.teams,
        ec.friends,
        ec.citations,
        ec.currency,
        ec.towns,
        ec.offices,
        ec.project,
        ec.causal,
        ec.konigsberg,
        ec.org_chart,
        ec.proteins,
        ec.food_web,
        ec.control_flow,
    ],
)
def test_loaded_as_one_step(loader):
    graph = loader()
    assert len(graph.steps) == 1
    assert graph.steps[0].code.startswith(loader.__name__ + "(")


class TestMetro:
    def test_attributes(self):
        metro = ec.metro()
        nodes = metro.get_node_df()
        assert set(nodes["zone"].to_list()) == {1, 2, 3}
        central = nodes.filter(nodes["label"] == "Central").row(0, named=True)
        assert central["lines"] == ["Red", "Blue"]
        assert central["pos"] == [0.0, 0.0]
        edges = metro.get_edge_df()
        assert set(edges["line"].to_list()) == {"Red", "Blue", "Green", "Ring", "Airport"}
        assert edges["minutes"].cast(float).to_list() == edges["weight"].to_list()

    def test_bridges(self):
        metro = ec.metro()
        assert edge_ends(metro, metro.bridges()) == {
            ("Harbor", "Riverside"),
            ("Docks", "Harbor"),
            ("Harbor", "Shipyard"),
            ("Business Park", "Eastfield"),
            ("Airport", "Business Park"),
        }

    def test_articulation_points(self):
        metro = ec.metro()
        aps = labels(metro, metro.articulation_points())
        assert aps == ["Business Park", "Eastfield", "Harbor", "Riverside"]
        assert "Central" not in aps

    def test_two_kinds_of_central(self):
        metro = ec.metro()
        assert top(metro.closeness_centrality(return_dict=True), metro) == "Central"
        assert top(metro.betweenness_centrality(return_dict=True), metro) == "Market"

    def test_fastest_is_not_fewest_stops(self):
        metro = ec.metro()
        fastest = metro.shortest_path("Westgate", "Lakeside", return_list=True)
        fewest = metro.astar_path("Westgate", "Lakeside", weighted=False)
        name = metro.get_label_from_id
        assert [name(i) for i in fastest] == [
            "Westgate",
            "Mill Lane",
            "Old Town",
            "Central",
            "Market",
            "Eastfield",
            "Lakeside",
        ]
        assert [name(i) for i in fewest] == [
            "Westgate",
            "Hillcrest",
            "Northpark",
            "Garden",
            "Lakeside",
        ]
        minutes = {tuple(sorted((u, v))): m for _, u, v, m in _METRO_SEGMENTS}

        def total(path):
            names = [name(i) for i in path]
            return sum(minutes[tuple(sorted(pair))] for pair in zip(names, names[1:]))

        assert total(fastest) == 18
        assert total(fewest) == 22

    def test_straight_line_never_overestimates(self):
        for _, u, v, minutes in _METRO_SEGMENTS:
            (x1, y1, _), (x2, y2, _) = _METRO_STATIONS[u], _METRO_STATIONS[v]
            assert math.dist((x1, y1), (x2, y2)) <= minutes

    def test_closing_stations(self):
        metro = ec.metro()
        curves = {
            strategy: [
                share
                for _, share in metro.network_robustness(
                    strategy=strategy, steps=5, seed=1, return_dict=True
                )
            ]
            for strategy in ["random", "degree", "betweenness"]
        }
        assert curves == {
            "random": pytest.approx([1.0, 0.95, 0.9, 0.85, 0.75, 0.7]),
            "degree": pytest.approx([1.0, 0.95, 0.9, 0.65, 0.5, 0.5]),
            "betweenness": pytest.approx([1.0, 0.8, 0.75, 0.6, 0.55, 0.55]),
        }
        # the attack counts stops, not minutes
        stops = metro.betweenness_centrality(weighted=False, return_dict=True)
        ranked = sorted(stops, key=lambda node: (-stops[node], node))
        closed = [metro.get_label_from_id(node) for node in ranked[:4]]
        assert closed == ["Riverside", "Market", "Eastfield", "Old Town"]
        damaged = ec.metro()
        damaged.delete_node("Riverside")
        cut_off = min(damaged.weakly_connected_components(), key=len)
        assert labels(damaged, cut_off) == ["Docks", "Harbor", "Shipyard"]
        assert len(metro.weakly_connected_components()) == 1
        for station in closed[1:]:
            damaged.delete_node(station)
        pieces = sorted(damaged.weakly_connected_components(), key=len, reverse=True)
        assert [len(piece) for piece in pieces] == [11, 3, 2]
        assert "Northpark" in labels(damaged, pieces[0])

    def test_every_way_to_the_airport(self):
        metro = ec.metro()
        routes = metro.all_simple_paths("Westgate", "Airport")
        assert len(routes) == 26
        assert sorted({len(route) - 1 for route in routes}) == list(range(7, 15))
        used = {frozenset(pair) for route in routes for pair in zip(route, route[1:])}
        assert len(used) == 22
        assert metro.count_edges() == 25
        unused = {
            tuple(sorted(metro.get_label_from_id(n) for n in pair))
            for pair in map(frozenset, metro.get_edge_df().select("source", "target").iter_rows())
            if pair not in used
        }
        assert unused == {("Harbor", "Riverside"), ("Docks", "Harbor"), ("Harbor", "Shipyard")}

    def test_layers_from_the_ends_and_from_central(self):
        metro = ec.metro()
        nodes = metro.get_node_df()
        ends = nodes.filter(nodes["degree"] == 1)["id"].to_list()
        assert labels(metro, ends) == ["Airport", "Docks", "Shipyard"]
        layers = metro.bfs_layers(ends)
        assert len(layers) == 7
        assert labels(metro, layers[-1]) == ["Hillcrest"]
        from_central = metro.bfs_layers("Central")
        assert len(from_central) == 5
        assert labels(metro, from_central[-1]) == ["Airport", "Garden", "Hillcrest"]

    def test_two_closures_split_the_core(self):
        metro = ec.metro()
        core = max(metro.biconnected_components(), key=len)
        assert len(core) == 15
        assert metro.node_connectivity() == 1
        assert metro.subgraph(core).node_connectivity() == 2
        split = metro.subgraph(core)
        split.delete_node("Old Town").delete_node("Northpark")
        pieces = sorted(split.weakly_connected_components(), key=len)
        assert len(pieces) == 2
        assert labels(split, pieces[0]) == ["Hillcrest", "Mill Lane", "Westgate"]

    def test_backup_routes_to_the_airport(self):
        metro = ec.metro()
        routes = metro.k_shortest_paths("Westgate", "Airport", k=4)
        assert [minutes for _, minutes in routes] == [23.0, 25.0, 26.0, 26.0]
        assert "Central" in labels(metro, routes[0][0])
        second = labels(metro, routes[1][0])
        assert "Riverside" in second and "Central" not in second

    def test_one_way_closure(self):
        lines = ec.metro().to_directed()
        assert lines.count_edges() == 50
        assert lines.shortest_path_length("Westgate", "Eastfield") == 14
        lines.delete_edge("Central", "Market")
        assert lines.shortest_path_length("Westgate", "Eastfield") == 16
        assert lines.shortest_path_length("Eastfield", "Westgate") == 14
        east = lines.shortest_path("Westgate", "Eastfield", return_list=True)
        assert "Riverside" in labels(lines, east)


class TestPayments:
    def test_includes_the_homepage_payments(self):
        homepage = [
            ("Dee", "Eve", 80),
            ("Eve", "Dee", 25),
            ("Ana", "Dee", 120),
            ("Dee", "Ben", 60),
            ("Ben", "Eve", 300),
            ("Gus", "Dee", 45),
            ("Eve", "Finn", 2000),
            ("Ana", "Ben", 9500),
            ("Ben", "Cruz", 9400),
            ("Cruz", "Ana", 9300),
            ("Finn", "Gus", 4950),
            ("Gus", "Hal", 4900),
            ("Hal", "Finn", 4850),
        ]
        payments = ec.payments()
        found = payments.match(
            "MATCH (a)-[p]->(b) RETURN a.name AS a, b.name AS b, p.amount AS amount"
        )
        assert set(homepage) <= set(found.iter_rows())

    def test_rings(self):
        payments = ec.payments()
        rings = payments.match("""
            MATCH (a)-[x]->(b)-[y]->(c)-[z]->(a)
            WHERE x.amount >= 1000 AND y.amount >= 1000 AND z.amount >= 1000
              AND a.name < b.name AND a.name < c.name
            RETURN a.name AS a, b.name AS b, c.name AS c, x.date AS starts
        """)
        assert sorted(rings.iter_rows()) == [
            ("Ana", "Ben", "Cruz", "2026-03-10"),
            ("Finn", "Gus", "Hal", "2026-03-16"),
        ]

    def test_pass_through_chain(self):
        payments = ec.payments()
        through = payments.match("""
            MATCH (a)-[x]->(b)-[y]->(c)
            WHERE x.amount >= 1000 AND y.amount < x.amount AND a <> c
            RETURN b.name AS account, x.amount AS received, y.amount AS sent
        """).filter(pl.col("received") - pl.col("sent") <= 100)
        rows = {row[0]: row[1:] for row in through.iter_rows()}
        # the chain, plus the accounts in the rings that passed a payment on
        assert set(rows) == {"Ivy", "Jo", "Kai", "Ben", "Cruz", "Gus", "Hal"}
        assert rows["Ivy"] == (3000, 2950)
        for name in ("Ivy", "Jo", "Kai"):
            assert payments.get_in_degree(name) == 1
            assert payments.get_out_degree(name) == 1

    def test_hub(self):
        payments = ec.payments()
        in_degrees = {
            name: payments.get_in_degree(name) for name in payments.get_node_df()["label"]
        }
        assert max(in_degrees, key=in_degrees.get) == "Grocer"
        assert in_degrees["Grocer"] == 8
        assert sorted(in_degrees.values())[-2] < 8
        assert payments.get_out_degree("Grocer") == 0

    def test_reciprocity(self):
        payments = ec.payments()
        assert payments.reciprocity() == pytest.approx(4 / 35)
        edges = payments.get_edge_df()
        pairs = set(zip(edges["source"], edges["target"]))
        mutual = {frozenset(labels(payments, p)) for p in pairs if p[::-1] in pairs}
        assert mutual == {frozenset({"Dee", "Eve"}), frozenset({"Max", "Nia"})}

    def test_strong_components(self):
        payments = ec.payments()
        big = [labels(payments, c) for c in payments.strongly_connected_components() if len(c) > 1]
        assert sorted(big) == [
            ["Ana", "Ben", "Cruz", "Dee", "Eve", "Finn", "Gus", "Hal"],
            ["Max", "Nia"],
        ]
        assert payments.is_weakly_connected()

    def test_one_payment_lets_money_into_finns_ring(self):
        payments = ec.payments()
        payments.delete_edge("Eve", "Finn")
        big = [labels(payments, c) for c in payments.strongly_connected_components() if len(c) > 1]
        assert sorted(big, key=len) == [
            ["Max", "Nia"],
            ["Finn", "Gus", "Hal"],
            ["Ana", "Ben", "Cruz", "Dee", "Eve"],
        ]

    def test_one_payment_lets_money_out_of_finns_ring(self):
        payments = ec.payments()
        payments.delete_edge("Gus", "Dee")
        big = [labels(payments, c) for c in payments.strongly_connected_components() if len(c) > 1]
        assert sorted(big, key=len) == [
            ["Max", "Nia"],
            ["Finn", "Gus", "Hal"],
            ["Ana", "Ben", "Cruz", "Dee", "Eve"],
        ]
        # Ana's payment back to Dee isn't the only way back from the first ring
        payments = ec.payments()
        payments.delete_edge("Ana", "Dee")
        assert max(len(c) for c in payments.strongly_connected_components()) == 8

    def test_following_the_money(self):
        payments = ec.payments()
        payments.select_nodes_by_id([payments.get_id_from_label("Dee")])
        first = payments.trav_out().get_selected_node_ids()
        second = payments.trav_out().get_selected_node_ids()
        assert labels(payments, first) == ["Ben", "Cafe", "Eve", "Grocer"]
        assert labels(payments, second) == ["Cafe", "Cruz", "Dee", "Eve", "Finn", "Grocer"]

        def is_business(node):
            return payments.get_node(node).attributes["kind"] == "business"

        cruz = payments.get_id_from_label("Cruz")
        payments.select_nodes_by_id([cruz], set_op="replace").trav_out_until(is_business)
        assert labels(payments, payments.get_selected_node_ids()) == ["Cafe", "Grocer"]
        payments.select_nodes_by_id([cruz], set_op="replace").trav_out_until(
            is_business, add_to_selection=True, exclude_unmatched=False
        )
        reached = set(labels(payments, payments.get_selected_node_ids()))
        assert len(reached) == 14
        assert reached.isdisjoint({"Acme", "Max", "Nia", "Landlord"})

    def test_only_max_pays_the_landlord(self):
        payments = ec.payments()
        assert payments.get_degree("Max") == 5
        assert labels(payments, payments.get_predecessors("Landlord")) == ["Max"]
        payments.delete_node("Max")
        assert payments.count_weakly_connected_components() == 2
        assert payments.get_degree("Landlord") == 0

    def test_acme_pays_four_wages(self):
        payments = ec.payments()
        assert labels(payments, payments.get_successors("Acme")) == ["Dee", "Eve", "Max", "Nia"]

    def test_payments_between_people(self):
        payments = ec.payments()
        payments.select_nodes(lambda node: node.attributes["kind"] == "business")
        payments.delete_nodes_ws()
        assert (payments.count_nodes(), payments.count_edges()) == (14, 19)

    def test_large_payments(self):
        payments = ec.payments()
        payments.select_edges(lambda edge: edge.weight < 1000)
        payments.delete_edges_ws()
        assert payments.count_edges() == 16
        alone = [
            label for label in payments.get_node_df()["label"] if payments.get_degree(label) == 0
        ]
        assert sorted(alone) == ["Cafe", "Grocer"]

    def test_undirected_dealings(self):
        payments = ec.payments()
        dealings = payments.to_undirected()
        assert dealings.count_edges() == 33
        kept = dealings.find_edge_ids("Dee", "Eve") + dealings.find_edge_ids("Max", "Nia")
        amounts = dealings.get_edge_df().filter(pl.col("id").is_in(kept))["amount"]
        assert sorted(amounts) == [80, 900]

    def test_payments_in_and_out(self):
        payments = ec.payments()
        businesses = ["Grocer", "Cafe", "Landlord", "Acme"]
        assert [payments.get_in_degree(b) for b in businesses] == [8, 3, 1, 0]
        assert payments.get_out_degree("Acme") == 4
        sinks = [n for n in payments.get_node_ids() if payments.get_out_degree(n) == 0]
        assert labels(payments, sinks) == ["Cafe", "Grocer", "Landlord"]

        def is_business(node):
            return node.attributes["kind"] == "business"

        def is_person(node):
            return not is_business(node)

        assert payments.aggregate_in_degree("mean", conditions=is_business) == 3.0
        assert payments.aggregate_in_degree("mean", conditions=is_person) == pytest.approx(23 / 14)
        assert payments.aggregate_out_degree("median", conditions=is_business) == 0.0
        assert payments.aggregate_out_degree("sum", conditions=is_business) == 4.0
        assert payments.aggregate_out_degree("mean", conditions=is_person) == pytest.approx(31 / 14)

    def test_grocer_pays_no_one(self):
        payments = ec.payments()
        dealings = payments.to_undirected()
        assert payments.is_directed() and not dealings.is_directed()
        assert payments.get_successors("Grocer") == []
        assert len(dealings.get_neighbors("Grocer")) == 8


class TestCourses:
    def test_is_a_dag(self):
        assert ec.courses().is_dag()

    def test_everything_in_cs_needs_cs101(self):
        courses = ec.courses()
        cs = {c for c in courses.get_node_df()["label"] if c.startswith("CS")}
        assert set(labels(courses, courses.descendants("CS101"))) == cs - {"CS101"}

    def test_longest_chain(self):
        courses = ec.courses()
        assert labels(courses, courses.dag_longest_path()) == sorted(
            ["MATH101", "MATH102", "MATH210", "STAT201", "CS360", "CS460"]
        )
        assert [courses.get_label_from_id(i) for i in courses.dag_longest_path()] == [
            "MATH101",
            "MATH102",
            "MATH210",
            "STAT201",
            "CS360",
            "CS460",
        ]
        assert len(courses.topological_generations()) == 6
        cs_only = courses.subgraph(
            [i for i in courses.get_node_ids() if courses.get_label_from_id(i).startswith("CS")]
        )
        assert cs_only.dag_longest_path_length() == 4

    def test_one_redundant_prerequisite(self):
        courses = ec.courses()
        reduced = courses.transitive_reduction()
        assert reduced.count_edges() == courses.count_edges() - 1
        kept = {
            (reduced.get_label_from_id(s), reduced.get_label_from_id(t))
            for s, t in reduced.get_edge_df().select("source", "target").iter_rows()
        }
        assert ("CS101", "CS201") not in kept
        assert ("CS102", "CS201") in kept
        assert sorted(reduced.get_node_ids()) == sorted(courses.get_node_ids())
        assert len(courses.all_simple_paths("CS101", "CS460")) == 2
        assert len(reduced.all_simple_paths("CS101", "CS460")) == 1

    def test_two_independent_tracks(self):
        courses = ec.courses()
        systems = ["CS210", "CS310", "CS320", "CS410"]
        data = ["CS250", "CS350", "CS360", "CS460"]
        for a, b in itertools.product(systems, data):
            assert b not in labels(courses, courses.descendants(a))
            assert a not in labels(courses, courses.descendants(b))
        assert (
            courses.get_label_from_id(courses.lowest_common_ancestor("CS410", "CS460")) == "CS102"
        )

    def test_prerequisites_and_next_courses(self):
        courses = ec.courses()
        assert labels(courses, courses.get_predecessors("CS360")) == ["CS201", "STAT201"]
        assert len(courses.ancestors("CS360")) == 8
        assert labels(courses, courses.get_successors("CS102")) == [
            "CS201",
            "CS210",
            "CS220",
            "CS250",
        ]
        ends = [c for c in courses.get_node_ids() if not courses.get_successors(c)]
        assert labels(courses, ends) == ["CS220", "CS350", "CS410", "CS460"]

    def test_prerequisite_loop(self):
        edited = ec.courses().add_edge("CS410", "CS210")
        assert not edited.is_dag()
        loops = sorted(labels(edited, cycle) for cycle in edited.simple_cycles())
        assert loops == [["CS210", "CS310", "CS410"], ["CS210", "CS320", "CS410"]]


class TestPipeline:
    def test_bottleneck(self):
        pipeline = ec.pipeline()
        value, (_, city_side) = pipeline.minimum_cut("Reservoir", "City")
        assert value == 45
        assert labels(pipeline, city_side) == [
            "City",
            "North District",
            "Ridge Tank",
            "South District",
            "Valley Junction",
        ]
        flow = pipeline.maximum_flow("Reservoir", "City")
        city = pipeline.get_id_from_label("City")
        into_city = sum(f for eid, f in flow.items() if pipeline.get_edge(eid).target == city)
        assert into_city == 45

    @pytest.mark.parametrize("demand, cost", [(10, 50), (35, 200), (40, 235), (45, 285)])
    def test_marginal_cost(self, demand, cost):
        total_flow, total_cost, _ = ec.pipeline().minimum_cost_flow(
            "Reservoir", "City", demand=demand
        )
        assert (total_flow, total_cost) == (demand, cost)

    def test_pumping_only_for_the_last_five(self):
        pipeline = ec.pipeline()
        pump = pipeline.get_id_from_label("Hill Pump")

        def pumped(demand):
            _, _, flow = pipeline.minimum_cost_flow("Reservoir", "City", demand=demand)
            return sum(f for eid, f in flow.items() if pipeline.get_edge(eid).source == pump)

        assert pumped(40) == 0
        assert pumped(45) == 5


class TestExams:
    def test_one_clique_of_four(self):
        exams = ec.exams()
        cliques = [labels(exams, c) for c in exams.find_cliques()]
        assert max(len(c) for c in cliques) == 4
        assert [c for c in cliques if len(c) == 4] == [
            ["Biology", "Calculus", "Chemistry", "Physics"]
        ]

    def test_slots(self):
        exams = ec.exams()

        def slots(strategy):
            coloring = exams.greedy_color(strategy=strategy)
            assert exams.is_valid_coloring(coloring)
            return len(set(coloring.values()))

        assert slots("largest_first") == 5
        assert slots("dsatur") == 4
        assert slots("smallest_last") == 4

    def test_compatible_exams(self):
        exams = ec.exams()
        compatible = exams.complement()
        assert compatible.count_edges() == 91 - 29
        assert labels(compatible, compatible.maximum_clique()) == [
            "History",
            "Literature",
            "Physics",
            "Programming",
            "Psychology",
            "Statistics",
            "Web Design",
        ]


class TestTeams:
    def test_bipartite(self):
        team = ec.teams()
        assert team.is_bipartite()
        sides = team.get_node_df()["side"].to_list()
        assert sides.count("worker") == 8 and sides.count("task") == 8

    def test_halls_condition_fails(self):
        team = ec.teams()
        assert len(team.bipartite_maximum_matching()) == 7
        blocked = {"Database", "DevOps", "Security"}
        able = {team.get_label_from_id(n) for task in blocked for n in team.get_neighbors(task)}
        assert able == {"Omar", "Kofi"}
        with pytest.raises(ValueError):
            team.bipartite_minimum_weight_full_matching()

    def test_contractor(self):
        team = ec.teams(contractor=True)
        assert team.count_nodes() == 17 and team.count_edges() == 23
        matched = team.bipartite_minimum_weight_full_matching()
        assert len(matched) == 8
        pairs = edge_ends(team, matched)
        assert ("DevOps", "Quinn") in pairs
        assert ("Kofi", "Security") in pairs
        assert ("Database", "Omar") in pairs
        assert not any("Luis" in pair for pair in pairs)
        assert sum(team.get_edge(e).weight for e in matched) == 37

    def test_cheapest_assignment_is_unique(self):
        from edgecase.example_networks import _TEAM_CONTRACTOR, _TEAM_SKILLS, _TEAM_TASKS

        hours = {(w, t): h for w, t, h in _TEAM_SKILLS + _TEAM_CONTRACTOR}
        workers = sorted({w for w, _ in hours})
        totals = []
        for chosen in itertools.permutations(workers, len(_TEAM_TASKS)):
            pairs = list(zip(chosen, _TEAM_TASKS))
            if all(p in hours for p in pairs):
                totals.append(sum(hours[p] for p in pairs))
        totals.sort()
        assert totals[:2] == [37, 38]

    def test_sessions_by_line_graph(self):
        team = ec.teams()
        assert team.get_degree("Omar") == 4 and team.get_degree("Testing") == 4
        assert max(team.get_degree(n) for n in team.get_node_ids()) == 4
        sessions = team.line_graph()
        hours = sessions.greedy_color()
        assert len(set(hours.values())) == 4
        edge_of = dict(sessions.get_node_df().select("id", "original_edge_id").iter_rows())
        assert team.is_valid_edge_coloring({edge_of[s]: h for s, h in hours.items()})


class TestFriends:
    def test_snapshots(self):
        first, second = ec.friends(), ec.friends(snapshot=2)
        assert second.count_edges() == first.count_edges() + 8
        old = edge_ends(first, first.get_edge_ids(limit=None))
        assert old <= edge_ends(second, second.get_edge_ids(limit=None))
        since = second.get_edge_df()["since"]
        assert (since == 2025).sum() == 8 and since.max() == 2025
        assert first.get_edge_df()["since"].max() == 2024

    def test_bad_snapshot(self):
        with pytest.raises(ValueError):
            ec.friends(snapshot=3)

    def test_four_circles(self):
        friends = ec.friends()
        communities = friends.cmty_louvain(return_dict=True)
        groups = {}
        for nid, c in communities.items():
            groups.setdefault(c, set()).add(friends.get_label_from_id(nid))
        assert len(groups) == 4
        clubs = dict(zip(friends.get_node_df()["label"], friends.get_node_df()["club"]))
        # each circle is all of one club, except that the couple sit together
        for members in groups.values():
            assert len({clubs[m] for m in members - {"Hugo", "Tara"}}) == 1

    def test_couple_has_top_dispersion(self):
        friends = ec.friends()
        scores = {
            pair: friends.dispersion(*pair)
            for pair in edge_ends(friends, friends.get_edge_ids(limit=None))
        }
        best = max(scores, key=scores.get)
        assert best == ("Hugo", "Tara")
        assert scores[best] == 1.5 and friends.dispersion("Tara", "Hugo") == 1.5
        assert sorted(scores.values())[-2] == 0.5
        shared = set(friends.get_neighbors("Hugo")) & set(friends.get_neighbors("Tara"))
        assert set(labels(friends, shared)) == {"Ben", "Jules", "Oona", "Wes"}
        clubs = friends.get_node_attrs("club")
        assert sorted(clubs[node] for node in shared) == ["Chess"] * 2 + ["Climbing"] * 2
        assert friends.dispersion("Hugo", "Tara", normalized=False) == 6.0

    def test_seeds_come_from_every_circle(self):
        friends = ec.friends().cmty_louvain(attr_name="circle")
        seeds = friends.influence_maximization(
            k=4, probability=0.2, n_simulations=500, seed=1, return_dict=True
        )
        assert [friends.get_label_from_id(node) for node, _ in seeds] == [
            "Tara",
            "Kemal",
            "Vik",
            "Dev",
        ]
        assert sum(gain for _, gain in seeds) == pytest.approx(9.6, abs=0.05)
        circle = friends.get_node_attrs("circle")
        assert len({circle[node] for node, _ in seeds}) == 4
        nodes = friends.get_node_df()
        popular = nodes.filter(nodes["degree"] == nodes["degree"].max())["label"].to_list()
        assert sorted(popular) == ["Ben", "Hugo", "Jules", "Oona", "Tara"]
        by_label = dict(zip(nodes["label"], nodes["circle"]))
        assert by_label["Tara"] == by_label["Oona"] == by_label["Hugo"]

    def test_new_ties_close_triangles(self):
        friends, later = ec.friends(), ec.friends(snapshot=2)
        new = edge_ends(later, later.get_edge_ids(limit=None)) - edge_ends(
            friends, friends.get_edge_ids(limit=None)
        )
        names = friends.get_node_df()["label"].to_list()
        existing = edge_ends(friends, friends.get_edge_ids(limit=None))
        candidates = [
            pair for pair in itertools.combinations(sorted(names), 2) if pair not in existing
        ]
        scores = friends.jaccard_coefficient(candidates)
        ranked = sorted(scores, key=lambda item: -item[2])
        name = friends.get_label_from_id
        top_eight = {tuple(sorted((name(u), name(v)))) for u, v, _ in ranked[:8]}
        assert len(top_eight & new) == 7
        assert ("Femi", "Vik") in new and ("Femi", "Vik") not in top_eight
        common = set(friends.get_neighbors("Femi")) & set(friends.get_neighbors("Vik"))
        assert not common

    def test_a_year_of_changes(self):
        friends, later = ec.friends(), ec.friends(snapshot=2)
        changes = friends.differential_edges(later)
        assert changes["removed"] == []
        assert len(changes["added"]) == 8
        assert len({person for pair in changes["added"] for person in pair}) == 16
        club = dict(zip(*friends.get_node_df().select("label", "club")))
        assert [pair for pair in changes["added"] if club[pair[0]] != club[pair[1]]] == [
            ("Femi", "Vik")
        ]
        assert friends.edge_overlap(later)["overlap"] == 43
        assert friends.edge_overlap(later)["union_size"] == 51
        result = friends.graph_edit_distance(later)
        assert result["distance"] == 8
        assert result["edge_insertions"] == 8

    def test_rosa_circle_changed_most(self):
        friends, later = ec.friends(), ec.friends(snapshot=2)
        rosa = friends.ego_graph("Rosa").differential_edges(later.ego_graph("Rosa"))
        assert rosa == {
            "added": [("Mara", "Nico"), ("Mara", "Pablo"), ("Nico", "Pablo"), ("Nico", "Rosa")],
            "removed": [],
        }
        kept = {}
        for person in friends.get_node_df()["label"]:
            overlap = friends.ego_graph(person).edge_overlap(later.ego_graph(person))
            kept[person] = round(overlap["jaccard"], 2)
        ranked = sorted(kept.items(), key=lambda item: item[1])
        assert ranked[:4] == [("Rosa", 0.43), ("Ada", 0.44), ("Mara", 0.44), ("Pablo", 0.44)]
        assert kept["Femi"] == 0.8
        assert min(kept.values()) == 0.43 and max(kept.values()) == 1.0

    def test_neighborhoods(self):
        friends = ec.friends()
        club = friends.get_node_attrs("club")
        hugo = friends.get_neighbors("Hugo")
        assert labels(friends, hugo) == ["Ben", "Jules", "Oona", "Tara", "Wes"]
        assert sorted(club[n] for n in hugo) == ["Chess"] * 2 + ["Climbing"] * 3

        shared = friends.get_common_neighbors("Hugo", "Tara")
        assert labels(friends, shared) == ["Ben", "Jules", "Oona", "Wes"]
        assert sorted(club[n] for n in shared) == ["Chess"] * 2 + ["Climbing"] * 2
        assert not any(friends.has_edge(u, v) for u in shared for v in shared if u != v)

        others = friends.get_non_neighbors("Lou")
        assert len(friends.get_neighbors("Lou")) == 3 and len(others) == 20
        common = {
            friends.get_label_from_id(n): len(friends.get_common_neighbors("Lou", n))
            for n in others
        }
        assert common.pop("Iris") == 3
        assert max(common.values()) == 1
        assert ec.friends(snapshot=2).has_edge("Lou", "Iris")

    @staticmethod
    def _y2026():
        return (
            ec.friends(snapshot=2)
            .delete_edge("Dev", "Femi")
            .delete_edge("Gwen", "Lou")
            .delete_edge("Rosa", "Sven")
        )

    def test_club_lists_union(self):
        friends = ec.friends()
        nodes = friends.get_node_df()

        def club(name):
            return friends.subgraph(nodes.filter(club=name)["id"].to_list())

        chess = club("Chess")
        assert (chess.count_nodes(), chess.count_edges()) == (12, 20)
        lists = chess.union(club("Climbing"))
        assert lists.count_nodes() == 24
        assert (lists.count_edges(), friends.count_edges() - lists.count_edges()) == (38, 5)

    def test_snapshot_set_operations(self):
        y2024, y2025, y2026 = ec.friends(), ec.friends(snapshot=2), self._y2026()
        lasted = y2024.intersection(y2026)
        assert (lasted.count_nodes(), lasted.count_edges()) == (24, 40)
        new = y2025.difference(y2024)
        assert new.count_nodes() == 16 and new.count_edges() == 8
        lost = y2025.difference(y2026)
        assert sorted(lost.get_edges_str_repr(use_labels=True)) == [
            "Dev--Femi",
            "Gwen--Lou",
            "Rosa--Sven",
        ]
        changes = y2024.symmetric_difference(y2026)
        assert (changes.count_nodes(), changes.count_edges()) == (16, 11)
        assert y2026.symmetric_difference(y2024).count_edges() == 11
        assert (y2026.difference(y2024).count_edges(), y2024.difference(y2026).count_edges()) == (
            8,
            3,
        )
        years = [y2024, y2025, y2026]
        assert ec.Graph.union_all(years).count_edges() == 51
        assert ec.Graph.intersection_all(years).count_edges() == 40
        # seven new ties close a triangle; Femi and Vik were six steps apart
        steps = {
            pair: y2024.shortest_path_length(*pair)
            for pair in (tuple(s.split("--")) for s in new.get_edges_str_repr(use_labels=True))
        }
        assert sorted(steps.values()) == [2] * 7 + [6]
        assert steps.get(("Femi", "Vik"), steps.get(("Vik", "Femi"))) == 6

    def test_club_update_compose(self):
        y2024 = ec.friends()
        update = ec.friends(snapshot=2).set_node_attrs("club", "Chess", nodes=["Vik"])
        vik = y2024.get_id_from_label("Vik")
        assert y2024.compose(update).get_node_attrs("club")[vik] == "Chess"
        assert y2024.union(update).get_node_attrs("club")[vik] == "Climbing"
        updates = [
            ec.friends(),
            ec.friends(snapshot=2).delete_edge("Dev", "Femi"),
            ec.friends().set_node_attrs("club", "Chess", nodes=["Vik"]),
        ]
        latest = ec.Graph.compose_all(updates)
        assert latest.count_edges() == 51
        assert latest.get_node_attrs("club")[latest.get_id_from_label("Vik")] == "Chess"

    def test_two_towns_and_a_coach(self):
        towns = ec.friends().disjoint_union(ec.friends())
        assert (towns.count_nodes(), towns.count_edges()) == (48, 86)
        assert towns.get_label_from_id(25) == "Ada_2"
        friends = ec.friends()
        climbers = friends.get_node_df().filter(club="Climbing")["id"].to_list()
        climbing = friends.subgraph(climbers)
        assert (climbing.count_nodes(), climbing.count_edges()) == (12, 18)
        team = climbing.full_join(ec.Graph().add_nodes(["Coach"]))
        assert (team.count_edges(), team.get_degree("Coach")) == (30, 12)

    def test_friends_of_friends(self):
        friends = ec.friends(snapshot=2)
        reach = friends.power(2)
        assert reach.count_edges() == 110
        assert (friends.get_degree("Ada"), reach.get_degree("Ada")) == (4, 7)

    def test_selected_subgraphs(self):
        friends = ec.friends()
        friends.select_nodes(lambda n: n.attributes["club"] == "Climbing")
        climbing = friends.transform_to_subgraph_ws()
        assert (climbing.count_nodes(), climbing.count_edges()) == (12, 18)
        friends.select_edges(lambda e: e.attributes["since"] >= 2022)
        recent = friends.transform_to_subgraph_ws()
        assert (recent.count_nodes(), recent.count_edges()) == (17, 10)

    def test_hugo_ego_graph(self):
        friends = ec.friends()
        hugo = friends.ego_graph("Hugo")
        assert (hugo.count_nodes(), hugo.count_edges()) == (6, 9)
        assert sorted(hugo.get_node_df()["club"]) == ["Chess"] * 3 + ["Climbing"] * 3
        assert friends.ego_graph("Hugo", radius=2).count_nodes() == 17

    def test_bridges_between_clubs(self):
        friends = ec.friends(snapshot=2)
        club = friends.get_node_attrs("club")
        cross = [e for e, s, t in friends.get_edges_as_tuples() if club[s] != club[t]]
        bridges = friends.edge_subgraph(cross)
        assert (bridges.count_nodes(), bridges.count_edges()) == (8, 6)
        assert friends.subgraph(bridges.get_node_ids()).count_edges() == 10


class TestCitations:
    def test_cites_only_older_papers(self):
        citations = ec.citations()
        assert citations.is_dag()
        years = citations.get_node_attrs("year")
        for eid in citations.get_edge_ids(limit=None):
            citing, cited = citations.get_node_ids_by_edge_id(eid)
            assert years[citing] > years[cited]
        # drawn as a timeline
        positions = citations.get_node_attrs("pos")
        assert all(positions[n][0] == years[n] for n in years)

    def test_fields(self):
        fields = ec.citations().get_node_df()["field"].value_counts().sort("field").rows()
        assert fields == [("learning", 7), ("networks", 15)]

    def test_seminal_paper(self):
        citations = ec.citations()
        in_degrees = {p: citations.get_in_degree(p) for p in citations.get_node_df()["label"]}
        assert max(in_degrees, key=in_degrees.get) == "Moreau 2000"
        assert in_degrees["Moreau 2000"] == 9
        assert sorted(in_degrees.values())[-2] < 9
        assert top(citations.authority_centrality(return_dict=True), citations) == "Moreau 2000"
        assert top(citations.pagerank(return_dict=True), citations) == "Moreau 2000"

    def test_survey_is_the_top_hub(self):
        citations = ec.citations()
        hubs = citations.hub_centrality(return_dict=True)
        assert top(hubs, citations) == "Hughes 2020"
        assert citations.get_out_degree("Hughes 2020") == 11
        assert citations.get_in_degree("Hughes 2020") == 1
        # by a wide margin: more than twice the next paper's score
        assert sorted(hubs.values())[-2] * 2 < max(hubs.values())

    def test_one_paper_joins_the_fields(self):
        citations = ec.citations()
        fields = citations.get_node_attrs("field")
        crossing = set()
        for eid in citations.get_edge_ids(limit=None):
            citing, cited = citations.get_node_ids_by_edge_id(eid)
            if fields[citing] != fields[cited]:
                crossing.add(citations.get_label_from_id(citing))
        assert crossing == {"Iyer 2014"}
        undirected = citations.to_undirected()
        assert labels(undirected, undirected.articulation_points()) == ["Iyer 2014"]
        betweenness = citations.betweenness_centrality(return_dict=True)
        assert top(betweenness, citations) == "Iyer 2014"

    def test_pagerank_sees_both_fields(self):
        citations = ec.citations().pagerank().authority_centrality()
        ranks = citations.get_node_df().with_columns(
            pl.col("pagerank", "authority_centrality").rank("min", descending=True)
        )
        nakamura = ranks.filter(pl.col("label") == "Nakamura 2001")
        assert nakamura["pagerank"].item() == 2
        assert nakamura["authority_centrality"].item() == 18

    def test_reversed_citations(self):
        citations = ec.citations()
        cited_by = citations.reverse()
        assert citations.get_out_degree("Moreau 2000") == 0
        assert len(cited_by.get_successors("Moreau 2000")) == 9
        assert len(cited_by.descendants("Moreau 2000")) == 14
        assert top(cited_by.pagerank(return_dict=True), cited_by) == "Osei 2021"
        assert "Hughes 2020" in labels(citations, citations.get_successors("Osei 2021"))


def rates(graph):
    """The quoted rates, keyed by (source label, target label)."""
    label = graph.get_label_from_id
    edges = graph.get_edge_df()
    return {
        (label(u), label(v)): r for u, v, r in zip(edges["source"], edges["target"], edges["rate"])
    }


class TestCurrency:
    def test_weights_are_negative_log_rates(self):
        edges = ec.currency().get_edge_df()
        for rate, weight, neg_log in zip(edges["rate"], edges["weight"], edges["neg_log_rate"]):
            assert weight == neg_log == pytest.approx(-math.log(rate))
        assert (edges["weight"] < 0).sum() > 0

    def test_every_pair_quoted_both_ways(self):
        quotes = rates(ec.currency())
        assert all((v, u) in quotes for u, v in quotes)

    def test_one_arbitrage_loop(self):
        fx = ec.currency()
        quotes = rates(fx)
        profitable = []
        for cycle in nx.simple_cycles(nx.DiGraph(list(quotes))):
            gain = math.prod(quotes[u, v] for u, v in zip(cycle, cycle[1:] + cycle[:1]))
            if gain > 1:
                profitable.append((cycle, gain))
        assert len(profitable) == 1
        cycle, gain = profitable[0]
        # dollars to euros, euros to pounds, pounds back to dollars
        start = cycle.index("USD")
        assert cycle[start:] + cycle[:start] == ["USD", "EUR", "GBP"]
        assert round(gain, 5) == 1.00099
        assert fx.negative_edge_cycle()
        loop = [fx.get_label_from_id(n) for n in fx.find_negative_cycle("USD")]
        assert set(loop) == {"USD", "EUR", "GBP"} and loop[0] == loop[-1] and len(loop) == 4

    def test_no_loop_without_arbitrage(self):
        fx = ec.currency(arbitrage=False)
        assert fx.steps[0].code == "currency(arbitrage=False)"
        assert not fx.negative_edge_cycle()
        changed = {pair for pair, rate in rates(fx).items() if rates(ec.currency())[pair] != rate}
        assert changed == {("EUR", "GBP"), ("GBP", "EUR")}

    def test_direct_route_is_not_the_best(self):
        fx = ec.currency(arbitrage=False)
        quotes = rates(fx)
        path = fx.bellman_ford_shortest_path("JPY", "CHF", return_list=True)
        assert [fx.get_label_from_id(n) for n in path] == ["JPY", "USD", "CHF"]
        via_usd = quotes["JPY", "USD"] * quotes["USD", "CHF"]
        assert via_usd > quotes["JPY", "CHF"]

    def test_hub_and_spoke(self):
        fx = ec.currency()
        assert fx.get_out_degree("USD") == 5
        assert fx.get_out_degree("CAD") == 1
        assert fx.get_in_degree("CAD") == 1


class TestTowns:
    @staticmethod
    def tour_length(towns, tour):
        return sum(towns.shortest_path_length(a, b) for a, b in zip(tour, tour[1:]))

    def test_shortest_round_trip(self):
        """Held-Karp over the road distances: the best tour is 396 km."""
        g = nx.Graph()
        g.add_weighted_edges_from(_ROADS)
        dist = dict(nx.floyd_warshall(g))
        names = list(_TOWNS)
        n = len(names)
        best = {(1, 0): 0}
        for mask in range(1, 1 << n, 2):
            for last in range(n):
                if (mask, last) not in best:
                    continue
                for nxt in range(n):
                    if not mask >> nxt & 1:
                        key = (mask | 1 << nxt, nxt)
                        cost = best[mask, last] + dist[names[last]][names[nxt]]
                        best[key] = min(best.get(key, math.inf), cost)
        full = (1 << n) - 1
        assert min(best[full, i] + dist[names[i]][names[0]] for i in range(1, n)) == 396

    def test_tsp_methods(self):
        towns = ec.towns()
        lengths = {
            method: self.tour_length(towns, towns.traveling_salesman(method=method))
            for method in ["greedy", "christofides"]
        }
        assert lengths == {"greedy": 533, "christofides": 429}
        best = towns.traveling_salesman(method="threshold_accepting", seed=1)
        assert self.tour_length(towns, best) == 396
        # the best tour visits each town once
        assert len(best) == 13 and len(set(best)) == 12
        # greedy doubles back from Dunholt to reach Millbrook, then comes home the way it came
        greedy = [towns.get_label_from_id(n) for n in towns.traveling_salesman(method="greedy")]
        assert greedy[greedy.index("Dunholt") :] == [
            "Dunholt", "Elmwick", "Fenmoor", "Millbrook", "Fenmoor", "Calder", "Bramford", "Ashby",
        ]  # fmt: skip
        assert greedy[:3] == ["Ashby", "Bramford", "Calder"]
        assert towns.get_label_from_id(best[-2]) == "Dunholt"

    def test_straight_line_never_overestimates(self):
        for u, v, km in _ROADS:
            (x1, y1, _), (x2, y2, _) = _TOWNS[u], _TOWNS[v]
            assert math.dist((x1, y1), (x2, y2)) < km

    def test_steiner_tree(self):
        towns = ec.towns()
        tree = towns.steiner_tree(["Ashby", "Elmwick", "Halston"])
        assert labels(tree, tree.get_node_ids()) == [
            "Ashby",
            "Bramford",
            "Calder",
            "Elmwick",
            "Halston",
        ]
        assert sum(tree.get_edge_df()["km"]) == 127

    def test_hubs_and_sizes(self):
        towns = ec.towns()
        degree = {towns.get_label_from_id(n): towns.get_degree(n) for n in towns.get_node_ids()}
        assert max(degree, key=degree.get) == "Greyford" and degree["Greyford"] == 7
        assert degree["Calder"] == 6
        assert max(_TOWNS, key=lambda town: _TOWNS[town][2]) == "Calder"

    def test_minimum_spanning_tree(self):
        towns = ec.towns()
        assert sum(towns.minimum_spanning_tree().get_edge_df()["km"]) == 333
        assert sum(km for _, _, km in _ROADS) == 965

    def test_calder_annexes_fenmoor(self):
        towns = ec.towns()
        assert towns.get_edge_df().filter(
            pl.col("id").is_in(towns.find_edge_ids("Calder", "Fenmoor"))
        )["km"].to_list() == [42]
        county = towns.contracted_edge("Calder", "Fenmoor", self_loops=False)
        assert county.count_nodes() == 11
        assert (county.get_degree("Calder"), county.get_degree("Greyford")) == (7, 6)

        def km(graph, a, b):
            edges = graph.get_edge_df()
            return edges.filter(pl.col("id").is_in(graph.find_edge_ids(a, b)))["km"].to_list()

        # Calder's own roads are kept, not Fenmoor's shorter ones
        assert (km(county, "Calder", "Elmwick"), km(county, "Calder", "Greyford")) == ([37], [38])
        assert (km(towns, "Fenmoor", "Elmwick"), km(towns, "Fenmoor", "Greyford")) == ([32], [28])

    def test_neighbors_of_a_similar_size(self):
        towns = ec.towns()
        similar = {
            town: labels(towns, towns.get_similar_neighbors(town, "population", tol_pct=(25, 25)))
            for town in towns.get_node_df()["label"]
        }
        assert {town: found for town, found in similar.items() if found} == {
            "Bramford": ["Halston"],
            "Halston": ["Bramford"],
            "Kestrel Bay": ["Greyford"],
        }

    def test_kilometers_or_roads(self):
        towns = ec.towns()
        assert towns.is_weighted()
        by_km = towns.shortest_path("Elmwick", "Kestrel Bay", return_list=True)
        by_roads = towns.shortest_path("Elmwick", "Kestrel Bay", return_list=True, weighted=False)
        label = towns.get_label_from_id
        assert [label(n) for n in by_km] == ["Elmwick", "Calder", "Halston", "Kestrel Bay"]
        assert [label(n) for n in by_roads] == ["Elmwick", "Greyford", "Kestrel Bay"]
        assert towns.shortest_path_length("Elmwick", "Kestrel Bay") == 99
        km = {
            tuple(sorted((s, t))): w
            for s, t, w in towns.get_edge_df(labels=True).select("source", "target", "km").rows()
        }
        assert km[("Elmwick", "Greyford")] + km[("Greyford", "Kestrel Bay")] == 101

    def test_not_planar(self):
        towns = ec.towns()
        planar, kuratowski = towns.check_planarity(counterexample=True)
        assert not planar
        assert (kuratowski.count_nodes(), kuratowski.count_edges()) == (8, 11)
        degrees = kuratowski.degrees(return_dict=True)
        assert labels(kuratowski, [n for n, d in degrees.items() if d == 2]) == [
            "Halston",
            "Larkhill",
        ]
        roads = edge_ends(kuratowski, kuratowski.get_edge_ids())
        assert {("Elmwick", "Greyford"), ("Calder", "Fenmoor")} <= roads

    def test_spanner_keeps_trips_within_stretch(self):
        towns = ec.towns()
        counts = [towns.transform_with_spanner(stretch=s).count_edges() for s in (1.5, 2, 3)]
        assert counts == [20, 15, 13]
        kept = towns.transform_with_spanner(stretch=1.5)
        worst = max(
            (
                kept.shortest_path_length(a, b, weighted=True)
                / towns.shortest_path_length(a, b, weighted=True),
                tuple(labels(towns, [a, b])),
            )
            for a, b in itertools.combinations(towns.get_node_ids(), 2)
        )
        assert worst[0] <= 1.5 and worst[1] == ("Bramford", "Halston")
        assert kept.shortest_path_length("Bramford", "Halston", weighted=True) == 64
        assert towns.shortest_path_length("Bramford", "Halston") == 47


class TestOffices:
    def test_floors_and_plan(self):
        offices = ec.offices()
        assert offices.is_connected() and offices.is_planar()
        nodes = offices.get_node_df()
        assert sorted(nodes["floor"].value_counts()["count"].to_list()) == [10, 10]
        # the second floor is drawn above the first
        for row in nodes.iter_rows(named=True):
            assert (row["pos"][1] >= 12) == (row["floor"] == 2)

    def test_halls_are_the_only_smallest_dominating_set(self):
        offices = ec.offices()
        ids = offices.get_node_ids()
        reach = {n: set(offices.get_neighbors(n)) | {n} for n in ids}

        def dominating(nodes):
            return set().union(*(reach[n] for n in nodes)) == set(ids)

        assert not any(dominating(c) for c in itertools.combinations(ids, 3))
        smallest = [c for c in itertools.combinations(ids, 4) if dominating(c)]
        halls = ["East Hall 1", "East Hall 2", "West Hall 1", "West Hall 2"]
        assert [labels(offices, c) for c in smallest] == [halls]
        assert labels(offices, offices.minimum_weighted_dominating_set()) == halls
        assert len(offices.dominating_set()) == 10

    def test_one_way_between_floors(self):
        offices = ec.offices()
        assert ("Stairs 1", "Stairs 2") in edge_ends(offices, offices.bridges())
        assert labels(offices, offices.articulation_points()) == [
            "East Hall 1",
            "East Hall 2",
            "Stairs 1",
            "Stairs 2",
            "West Hall 1",
            "West Hall 2",
        ]

    def test_three_colors(self):
        offices = ec.offices()
        triangle = [offices.get_id_from_label(r) for r in ["Lobby", "Reception", "West Hall 1"]]
        for a, b in itertools.combinations(triangle, 2):
            assert b in offices.get_neighbors(a)
        assert len(set(offices.greedy_color(strategy="dsatur").values())) == 3

    def test_independent_set(self):
        offices = ec.offices()
        independent = offices.maximum_independent_set()
        assert len(independent) == 11
        assert offices.is_independent_set(independent)

    def test_faces(self):
        offices = ec.offices()
        faces = [[offices.get_label_from_id(n) for n in f] for f in offices.planar_faces()]
        assert offices.count_nodes() - offices.count_edges() + len(faces) == 2
        assert sorted(len(face) for face in faces) == [3, 3, 8, 9, 23]
        triangles = sorted(sorted(face) for face in faces if len(face) == 3)
        assert triangles == [
            ["East Hall 2", "Office 203", "Office 204"],
            ["Lobby", "Reception", "West Hall 1"],
        ]
        assert offices.count_edges() - len(offices.bridges()) == 14


class TestProject:
    @staticmethod
    def slack(project):
        """Each task's slack, from the earliest and latest it can start."""
        order = [n for stage in project.topological_generations() for n in stage]
        days = {n: project.get_node(n).attributes["days"] for n in order}
        earliest = {}
        for n in order:
            earliest[n] = max(
                (earliest[p] + days[p] for p in project.get_predecessors(n)), default=0
            )
        end = earliest[project.get_id_from_label("Finish")]
        latest = {}
        for n in reversed(order):
            latest[n] = min((latest[s] for s in project.get_successors(n)), default=end) - days[n]
        return end, {project.get_label_from_id(n): latest[n] - earliest[n] for n in order}

    def test_critical_path(self):
        project = ec.project()
        assert project.is_dag()
        assert [project.get_label_from_id(n) for n in project.dag_longest_path()] == [
            "Start",
            "Sign lease",
            "Design",
            "Permits",
            "Build out",
            "Install",
            "Training",
            "Soft opening",
            "Finish",
        ]
        assert project.dag_longest_path_length() == 66

    def test_slack(self):
        end, slack = self.slack(ec.project())
        assert end == 66
        assert slack["Order equipment"] == 20
        assert slack["Health inspection"] == 2
        assert slack["Hire staff"] == 42
        critical = {task for task, s in slack.items() if s == 0}
        assert critical == {
            "Start",
            "Sign lease",
            "Design",
            "Permits",
            "Build out",
            "Install",
            "Training",
            "Soft opening",
            "Finish",
        }

    def test_edges_weighted_by_the_first_task(self):
        project = ec.project()
        for row in project.get_edge_df().iter_rows(named=True):
            source = project.get_label_from_id(row["source"])
            assert row["weight"] == row["days"] == _TASKS[source][0]

    def test_stages_and_dominators(self):
        project = ec.project()
        assert len(project.topological_generations()) == 9
        label = project.get_label_from_id
        idom = {label(k): label(v) for k, v in project.immediate_dominators("Start").items()}
        assert idom["Install"] == "Design"
        assert idom["Soft opening"] == "Sign lease"


class TestCausal:
    def test_collider(self):
        causal = ec.causal()
        assert causal.is_d_separator(["Exercise"], ["Diet"], [])
        assert not causal.is_d_separator(["Exercise"], ["Diet"], ["Weight"])

    def test_confounder(self):
        """Age is the smallest adjustment set for the effect of Exercise on Blood pressure."""
        backdoor = ec.Graph.from_edges([e for e in _CAUSES if e[0] != "Exercise"], directed=True)
        found = backdoor.find_minimal_d_separator(["Exercise"], ["Blood pressure"])
        assert labels(backdoor, found) == ["Age"]

    def test_mediator(self):
        g = nx.DiGraph(_CAUSES)
        paths = list(nx.all_simple_paths(g, "Exercise", "Blood pressure"))
        assert paths and all("Weight" in path for path in paths)

    def test_unobserved_confounder(self):
        causal = ec.causal()
        observed = [n for n in causal.get_node_ids() if causal.get_node(n).attributes["observed"]]
        assert labels(causal, set(causal.get_node_ids()) - set(observed)) == ["Genetics"]
        backdoor = ec.Graph.from_edges(
            [e for e in _CAUSES if e[0] != "Blood pressure"], directed=True
        )
        restricted = ["Age", "Exercise", "Diet", "Weight", "Medication"]
        assert (
            backdoor.find_minimal_d_separator(
                ["Blood pressure"], ["Heart disease"], restricted=restricted
            )
            is None
        )

    def test_colliders_and_v_structures(self):
        causal = ec.causal()
        label = causal.get_label_from_id
        colliders = {tuple(map(label, c)) for c in causal.colliders()}
        v_structures = {tuple(map(label, c)) for c in causal.v_structures()}
        assert len(colliders) == 8 and len(v_structures) == 6
        assert v_structures < colliders
        shielded = colliders - v_structures
        assert ("Age", "Blood pressure", "Medication") in shielded


class TestMolecules:
    def test_shapes(self):
        molecules = ec.molecules()
        sizes = {name: (g.count_nodes(), g.count_edges()) for name, g in molecules.items()}
        assert sizes == {
            "benzene": (6, 6),
            "cyclohexane": (6, 6),
            "ethanol": (3, 2),
            "dimethyl_ether": (3, 2),
            "butane": (4, 3),
            "isobutane": (4, 3),
            "caffeine": (14, 15),
        }
        for name, g in molecules.items():
            assert not g.is_directed() and not g.is_weighted()
            assert g.is_connected()
            assert len(g.steps) == 1 and g.steps[0].code == f"molecules()[{name!r}]"
        elements = molecules["caffeine"].get_node_df()["element"].value_counts()
        assert dict(zip(elements["element"], elements["count"])) == {"C": 8, "N": 4, "O": 2}

    def test_same_shape_different_molecule(self):
        molecules = ec.molecules()
        ethanol, ether = molecules["ethanol"], molecules["dimethyl_ether"]
        assert ethanol.is_isomorphic(ether)
        assert not ethanol.is_isomorphic(ether, node_attr="element")
        benzene, cyclohexane = molecules["benzene"], molecules["cyclohexane"]
        assert benzene.is_isomorphic(cyclohexane)
        assert benzene.is_isomorphic(cyclohexane, node_attr="element")
        assert not benzene.is_isomorphic(cyclohexane, edge_attr="bond")
        assert not benzene.is_isomorphic(cyclohexane, node_attr="element", edge_attr="bond")

    def test_same_atoms_different_shape(self):
        molecules = ec.molecules()
        assert not molecules["butane"].is_isomorphic(molecules["isobutane"])

    def test_wiener_index(self):
        molecules = ec.molecules()
        assert molecules["butane"].wiener_index() == 10
        assert molecules["isobutane"].wiener_index() == 9

    def test_caffeine_rings(self):
        caffeine = ec.molecules()["caffeine"]
        assert sorted(map(len, caffeine.minimum_cycle_basis())) == [5, 6]
        assert len(caffeine.cycle_basis()) == 2
        assert caffeine.girth() == 5

    @staticmethod
    def _purine_caffeine():
        # Caffeine with its ring atoms numbered by the scheme for purines, as in the
        # isomorphism docstrings
        purine = ec.Graph.from_edges(
            [("N1", "C2"), ("C2", "N3"), ("N3", "C4"), ("C4", "C5"), ("C5", "C6"),
             ("C6", "N1"), ("C4", "N9"), ("N9", "C8"), ("C8", "N7"), ("N7", "C5"),
             ("C2", "O10"), ("C6", "O11"), ("N1", "C12"), ("N3", "C13"), ("N7", "C14")]
        )  # fmt: skip
        purine.set_node_attrs("element", {a: a[0] for a in purine.get_node_df()["label"]})
        purine.set_edge_attrs("bond", "single")
        purine.set_edge_attrs(
            "bond", "double", edges=[("C4", "C5"), ("N9", "C8"), ("C2", "O10"), ("C6", "O11")]
        )
        return purine

    def test_caffeine_numberings_match(self):
        caffeine = ec.molecules()["caffeine"]
        purine = self._purine_caffeine()
        mapping = caffeine.find_isomorphism(purine, node_attr="element", edge_attr="bond")
        named = {
            caffeine.get_label_from_id(k): purine.get_label_from_id(v) for k, v in mapping.items()
        }
        assert named == {
            "C1": "C14", "N1": "N7", "C2": "C8", "N2": "N9", "C3": "C4", "C4": "C5",
            "C5": "C6", "O1": "O11", "N3": "N1", "C6": "C2", "O2": "O10", "N4": "N3",
            "C7": "C13", "C8": "C12",
        }  # fmt: skip
        assert len(caffeine.subgraph_isomorphisms(purine, node_attr="element")) == 1

        def form(graph):
            number = graph.canonical_labeling(node_attr="element")
            return sorted(
                tuple(sorted((number[s], number[t]))) for _, s, t in graph.get_edges_as_tuples()
            )

        assert form(caffeine) == form(purine)
        c1, c14 = caffeine.get_id_from_label("C1"), purine.get_id_from_label("C14")
        assert caffeine.canonical_labeling(node_attr="element")[c1] == 3
        assert purine.canonical_labeling(node_attr="element")[c14] == 3

    def test_ethanol_maps_onto_ether_by_shape_only(self):
        molecules = ec.molecules()
        ethanol, ether = molecules["ethanol"], molecules["dimethyl_ether"]
        mapping = ethanol.find_isomorphism(ether)
        named = {
            ethanol.get_label_from_id(k): ether.get_label_from_id(v) for k, v in mapping.items()
        }
        assert named == {"C1": "C1", "C2": "O1", "O1": "C2"}
        assert ethanol.find_isomorphism(ether, node_attr="element") is None

    def test_caffeine_carbonyls_and_rings(self):
        caffeine = ec.molecules()["caffeine"]
        label = caffeine.get_label_from_id
        carbonyl = ec.Graph.from_edges([("C", "O")])
        carbonyl.set_node_attrs("element", {"C": "C", "O": "O"})
        carbonyl.set_edge_attrs("bond", {1: "double"})
        matches = caffeine.subgraph_isomorphisms(carbonyl, node_attr="element", edge_attr="bond")
        assert sorted(tuple(map(label, m.values())) for m in matches) == [
            ("C5", "O1"),
            ("C6", "O2"),
        ]
        rings = {
            n: caffeine.subgraph_isomorphisms(ec.Graph().add_cycle_graph(n), symmetry=True)
            for n in [5, 6, 9]
        }
        assert {n: len(found) for n, found in rings.items()} == {5: 1, 6: 1, 9: 0}
        outer = ec.Graph().add_cycle_graph(9)
        assert len(caffeine.subgraph_isomorphisms(outer, induced=False, symmetry=True)) == 1
        five = set(map(label, rings[5][0].values()))
        six = set(map(label, rings[6][0].values()))
        assert five & six == {"C3", "C4"}
        assert six == {"C3", "C4", "C5", "C6", "N3", "N4"}

    def test_screening_pairs(self):
        molecules = ec.molecules()
        passing = [
            (a, b)
            for a, b in itertools.combinations(molecules, 2)
            if molecules[a].could_be_isomorphic(molecules[b])
        ]
        assert passing == [("benzene", "cyclohexane"), ("ethanol", "dimethyl_ether")]
        assert all(molecules[a].is_isomorphic(molecules[b]) for a, b in passing)
        assert not molecules["butane"].could_be_isomorphic(molecules["isobutane"])

    def test_benzene_ring_search(self):
        molecules = ec.molecules()
        benzene = molecules["benzene"]

        def containing(**kwargs):
            return {n for n, g in molecules.items() if g.is_subgraph_isomorphic(benzene, **kwargs)}

        assert containing() == {"benzene", "cyclohexane", "caffeine"}
        assert containing(node_attr="element") == {"benzene", "cyclohexane"}
        assert containing(node_attr="element", edge_attr="bond") == {"benzene"}

    def test_common_subgraph_with_benzene(self):
        molecules = ec.molecules()
        caffeine, benzene = molecules["caffeine"], molecules["benzene"]
        found = caffeine.largest_common_subgraph(benzene)
        assert len(found) == 1
        assert sorted(map(caffeine.get_label_from_id, found[0].values())) == [
            "C3", "C4", "C5", "C6", "N3", "N4"
        ]  # fmt: skip
        found = caffeine.largest_common_subgraph(benzene, node_attr="element")
        assert len(found) == 7
        carbons = list(found[0].values())
        assert [caffeine.get_label_from_id(n) for n in carbons] == ["C3", "C4", "C5", "C6"]
        bonds = {
            tuple(sorted(map(caffeine.get_label_from_id, (u, v))))
            for _, u, v in caffeine.get_edges_as_tuples()
            if u in carbons and v in carbons
        }
        assert bonds == {("C3", "C4"), ("C4", "C5")}

    def test_symmetry_counts(self):
        molecules = ec.molecules()

        def symmetries(g, **kwargs):
            return math.prod(len(c) for c in g.automorphism_cosets(**kwargs).values())

        counts = {
            n: (symmetries(g), symmetries(g, node_attr="element")) for n, g in molecules.items()
        }
        assert counts == {
            "benzene": (12, 12),
            "cyclohexane": (12, 12),
            "ethanol": (2, 1),
            "dimethyl_ether": (2, 2),
            "butane": (2, 2),
            "isobutane": (6, 6),
            "caffeine": (1, 1),
        }
        isobutane = molecules["isobutane"]
        coset = isobutane.automorphism_cosets()[isobutane.get_id_from_label("C1")]
        assert sorted(map(isobutane.get_label_from_id, coset)) == ["C1", "C3", "C4"]

    def test_hash_finds_duplicates(self):
        molecules = ec.molecules()
        molecules["caffeine_purine"] = self._purine_caffeine()

        def duplicates(**kwargs):
            groups = {}
            for name, g in molecules.items():
                groups.setdefault(g.weisfeiler_lehman_graph_hash(**kwargs), []).append(name)
            return [names for names in groups.values() if len(names) > 1]

        assert duplicates() == [
            ["benzene", "cyclohexane"],
            ["ethanol", "dimethyl_ether"],
            ["caffeine", "caffeine_purine"],
        ]
        assert duplicates(node_attr="element") == [
            ["benzene", "cyclohexane"],
            ["caffeine", "caffeine_purine"],
        ]
        assert duplicates(node_attr="element", edge_attr="bond") == [
            ["caffeine", "caffeine_purine"]
        ]

    def test_methyl_carbons_look_alike(self):
        caffeine = ec.molecules()["caffeine"]
        hashes = caffeine.weisfeiler_lehman_subgraph_hashes(node_attr="element")
        assert [len({h[i] for h in hashes.values()}) for i in range(3)] == [9, 12, 14]
        label = caffeine.get_label_from_id

        def shared(i):
            values = [h[i] for h in hashes.values()]
            return {label(n) for n, h in hashes.items() if values.count(h[i]) > 1}

        assert shared(1) == {"C1", "C7", "C8"}
        assert shared(0) == {"C1", "C7", "C8", "N1", "N3", "N4", "O1", "O2"}

    @staticmethod
    def _same_element(x, y):
        return x["element"] == y["element"]

    def test_caffeine_relatives_are_two_edits_apart(self):
        caffeine = ec.molecules()["caffeine"]
        relatives = [caffeine] + [
            ec.molecules()["caffeine"].delete_node(methyl) for methyl in ["C1", "C7", "C8"]
        ]
        for a, b in itertools.combinations(relatives, 2):
            result = a.graph_edit_distance(b, exact=True, node_match=self._same_element)
            assert result["distance"] == 2

    def test_butane_to_isobutane_paths(self):
        molecules = ec.molecules()
        butane, isobutane = molecules["butane"], molecules["isobutane"]
        paths, cost = butane.optimal_edit_paths(isobutane)
        assert (len(paths), cost) == (12, 2.0)

        def ends(graph, edge):
            return tuple(map(graph.get_label_from_id, graph.get_node_ids_by_edge_id(edge)))

        for node_path, edge_path in paths:
            deleted = [e for e, f in edge_path if f is None]
            inserted = [f for e, f in edge_path if e is None]
            assert len(deleted) == len(inserted) == 1
        node_path, edge_path = paths[0]
        assert all(
            butane.get_label_from_id(u) == isobutane.get_label_from_id(v) for u, v in node_path
        )
        assert [ends(butane, e) for e, f in edge_path if f is None] == [("C3", "C4")]
        assert [ends(isobutane, f) for e, f in edge_path if e is None] == [("C2", "C4")]

    def test_caffeine_to_benzene_estimates(self):
        molecules = ec.molecules()
        caffeine, benzene = molecules["caffeine"], molecules["benzene"]
        assert list(caffeine.optimize_graph_edit_distance(benzene)) == [21, 19, 17]
        by_element = list(
            caffeine.optimize_graph_edit_distance(benzene, node_match=self._same_element)
        )
        assert by_element == [27, 26, 24, 23, 22, 21, 19]
        ring = caffeine.subgraph_isomorphisms(benzene, limit=1)[0].values()
        elements = caffeine.get_node_attrs("element")
        assert sorted(elements[n] for n in ring) == ["C", "C", "C", "C", "N", "N"]

    def test_finding_the_missing_methyl(self):
        caffeine = ec.molecules()["caffeine"]
        theobromine = ec.molecules()["caffeine"].delete_node("C8")
        found = list(caffeine.optimize_edit_paths(theobromine, node_match=self._same_element))
        assert [cost for _, _, cost in found] == [4.0, 2.0]
        node_path, edge_path, _ = found[-1]
        label = caffeine.get_label_from_id
        assert [label(u) for u, v in node_path if v is None] == ["C8"]
        assert [
            tuple(map(label, caffeine.get_node_ids_by_edge_id(e)))
            for e, f in edge_path
            if f is None
        ] == [("N3", "C8")]
        rooted = caffeine.optimize_edit_paths(
            theobromine, node_match=self._same_element, roots=("C8", "C1")
        )
        assert [cost for _, _, cost in rooted] == [22, 20, 18, 14, 10, 4]

    def test_regular_rings(self):
        molecules = ec.molecules()
        assert molecules["benzene"].is_k_regular(2)
        assert not molecules["caffeine"].is_regular()
        degrees = molecules["caffeine"].degrees(return_dict=True).values()
        assert sorted(set(degrees)) == [1, 2, 3]


class TestKonigsberg:
    def test_every_node_is_odd(self):
        k = ec.konigsberg()
        degree = {k.get_label_from_id(n): k.get_degree(n) for n in k.get_node_ids()}
        assert degree == {"Kneiphof": 5, "North Bank": 3, "South Bank": 3, "Lomse": 3}
        assert not k.is_eulerian() and not k.has_eulerian_path()
        assert not k.is_simple()

    def test_two_more_bridges(self):
        k = ec.konigsberg()
        walkable = k.eulerize()
        assert walkable.count_edges() == 9 and walkable.is_eulerian()
        added = walkable.get_edge_df().filter(pl.col("eulerize_added"))
        assert edge_ends(walkable, added["id"].to_list()) == {
            ("Kneiphof", "North Bank"),
            ("Lomse", "South Bank"),
        }
        assert "Hohe Brücke" in added["bridge"].to_list()


class TestOrgChart:
    def test_tree(self):
        org = ec.org_chart()
        assert org.is_arborescence()
        assert [org.get_label_from_id(n) for n in org.dag_longest_path()] == [
            "Mira",
            "Jonas",
            "Ines",
            "Kwame",
            "Ruth",
            "Tariq",
            "Noor",
        ]
        assert sorted(set(org.get_node_df()["level"])) == list(range(7))

    def test_wide_team(self):
        org = ec.org_chart()
        reports = {org.get_label_from_id(n): org.get_out_degree(n) for n in org.get_node_ids()}
        assert max(reports, key=reports.get) == "Sam" and reports["Sam"] == 6
        assert sorted(reports.values())[-2] < 6

    def test_lowest_common_ancestors(self):
        org = ec.org_chart()
        label = org.get_label_from_id
        assert label(org.lowest_common_ancestor("Noor", "Pia")) == "Ruth"
        assert label(org.lowest_common_ancestor("Noor", "Ada")) == "Mira"

    def test_the_top_is_not_the_middle(self):
        undirected = ec.org_chart().to_undirected()
        assert labels(undirected, undirected.center()) == ["Ines", "Jonas"]
        betweenness = undirected.betweenness_centrality(return_dict=True)
        assert top(betweenness, undirected) == "Mira"

    def test_managers_drawn_over_reports(self):
        org = ec.org_chart()
        pos = {n: org.get_node(n).attributes["pos"] for n in org.get_node_ids()}
        for n in org.get_node_ids():
            reports = org.get_successors(n)
            if reports:
                xs = [pos[r][0] for r in reports]
                assert pos[n][0] == (min(xs) + max(xs)) / 2
                assert all(pos[r][1] == pos[n][1] - 1 for r in reports)

    def test_departments(self):
        nodes = ec.org_chart().get_node_df()
        widths = nodes.group_by("dept", "level").len().group_by("dept").agg(pl.col("len").max())
        depths = nodes.group_by("dept").agg(pl.col("level").n_unique())
        assert widths.sort("len")["dept"].to_list()[-1] == "Sales"
        assert widths.sort("len")["len"].to_list()[-2:] == [2, 6]
        assert depths.sort("level")["dept"].to_list()[-1] == "Engineering"

    def test_trees_forests_and_branchings(self):
        undirected = ec.org_chart().to_undirected()
        assert undirected.is_tree()
        tied = ec.org_chart().to_undirected().add_edge("Ruth", "Owen")
        assert not tied.is_tree()
        assert [labels(tied, c) for c in tied.cycle_basis()] == [["Kwame", "Owen", "Ruth"]]

        dual = ec.org_chart().add_edge("Jonas", "Ada")
        assert not dual.is_arborescence() and dual.is_dag()
        assert dual.get_in_degree("Ada") == 2

        headless = ec.org_chart().delete_node("Mira")
        assert headless.is_branching() and not headless.is_arborescence()
        roots = [n for n in headless.get_node_ids() if headless.get_in_degree(n) == 0]
        assert labels(headless, roots) == ["Jonas", "Leila", "Sam", "Uma"]
        forest = ec.org_chart().to_undirected().delete_node("Mira")
        assert forest.is_forest() and not forest.is_tree()
        assert forest.count_weakly_connected_components() == 4
        assert not headless.add_edge("Hiro", "Owen").is_branching()


class TestProteins:
    SEEDS = ["REP1", "REP2", "REP5", "CYC2", "TRN3"]

    def test_attributes(self):
        proteins = ec.proteins()
        df = proteins.get_node_df()
        assert df.filter(pl.col("disease_gene"))["label"].sort().to_list() == sorted(self.SEEDS)
        unannotated = df.filter(pl.col("pathway").is_null())["label"].to_list()
        assert unannotated == ["ORF1", "ORF2", "ORF3"]
        targets = df.filter(pl.col("drug_target").is_not_null())
        assert dict(zip(targets["label"], targets["drug_target"])) == {
            "REP6": "Drug A",
            "CYC3": "Drug A",
            "SIG4": "Drug B",
            "ENZ2": "Drug B",
        }
        assert proteins.is_connected()

    def test_drawing_has_no_crossings(self):
        proteins = ec.proteins()
        pos = {n: tuple(p) for n, p in proteins.get_node_attrs("pos").items()}
        assert_crossing_free(proteins, pos)

    def test_disease_module(self):
        proteins = ec.proteins()
        module = proteins.expand_seed_set(self.SEEDS[:4], max_size=12, min_score=0.5)
        expected = ["CYC2", "ORF1", "REP1", "REP2", "REP3", "REP4", "REP5", "REP6"]
        assert labels(proteins, module) == expected
        result = proteins.subgraph_enrichment(module, "pathway", "DNA repair")
        assert (result["k"], result["K"], result["n"], result["N"]) == (6, 6, 8, 31)
        assert result["p_value"] == pytest.approx(3.8e-5, rel=0.01)

    def test_lone_disease_gene(self):
        proteins = ec.proteins()
        prizes = {s: 3.0 for s in self.SEEDS}
        tree = proteins.prize_collecting_steiner_tree(prizes)
        assert sorted(tree.get_node_df()["label"]) == ["CYC2", "REP1", "REP2", "REP3", "REP5"]
        cheaper = proteins.prize_collecting_steiner_tree(prizes, edge_penalty=0.5)
        assert {"TRN3", "HSP1"} <= set(cheaper.get_node_df()["label"])
        assert proteins.shortest_path_length("TRN3", "REP1") == 3

    def test_sticky_hub(self):
        proteins = ec.proteins()
        degree = {
            proteins.get_label_from_id(n): len(proteins.get_neighbors(n))
            for n in proteins.get_node_ids()
        }
        assert degree["HSP1"] == 9 and sorted(degree.values())[-2] == 5
        rwr = proteins.random_walk_with_restart({s: 1.0 for s in self.SEEDS}, return_dict=True)
        others = sorted(
            (n for n in rwr if proteins.get_label_from_id(n) not in self.SEEDS),
            key=rwr.get,
            reverse=True,
        )
        assert labels(proteins, others[:2]) == ["HSP1", "REP4"]
        assert proteins.get_label_from_id(others[0]) == "REP4"
        module = proteins.expand_seed_set(self.SEEDS[:4], max_size=12, min_score=0.5)
        assert proteins.get_id_from_label("HSP1") not in module

    def test_smoothing(self):
        proteins = ec.proteins()
        smoothed = proteins.network_smooth(proteins.get_node_attrs("expression"), return_dict=True)
        by_label = {proteins.get_label_from_id(n): v for n, v in smoothed.items()}
        assert by_label["REP3"] == pytest.approx(1.08, abs=0.005)
        assert by_label["ENZ3"] == pytest.approx(1.35, abs=0.005)
        ranked = sorted(by_label, key=by_label.get, reverse=True)
        assert ranked[:6] == ["REP5", "REP4", "REP2", "REP1", "REP6", "CYC2"]
        assert ranked[6] == "ENZ3"
        raw = proteins.get_node_df().sort("expression", descending=True)["label"].to_list()
        assert raw[:2] == ["REP5", "ENZ3"]

    def test_uncharacterized_proteins(self):
        proteins = ec.proteins()
        pathways = proteins.get_node_attrs("pathway")
        names = sorted(set(pathways.values()))
        known = {n: names.index(p) for n, p in pathways.items()}
        result = proteins.label_propagation(known, return_dict=True)
        assert {
            proteins.get_label_from_id(n): names[c] for n, c in result.items() if n not in known
        } == {"ORF1": "DNA repair", "ORF2": "Signaling", "ORF3": "Transport"}

    def test_drug_proximity(self):
        proteins = ec.proteins()
        drug_a, drug_b = ["REP6", "CYC3"], ["SIG4", "ENZ2"]
        closest = [proteins.network_proximity(d, self.SEEDS)["proximity"] for d in (drug_a, drug_b)]
        assert closest == [1.0, 3.0]
        separation = [
            proteins.network_proximity(d, self.SEEDS, method="separation")["separation"]
            for d in (drug_a, drug_b)
        ]
        assert separation == pytest.approx([-0.05, 0.25])
        means = [
            [
                proteins.network_proximity(d, self.SEEDS, method=m)["proximity"]
                for d in (drug_a, drug_b)
            ]
            for m in ("shortest", "kernel")
        ]
        assert means[0] == pytest.approx([2.3, 3.6])
        assert means[1] == pytest.approx([0.152, 0.035], abs=5e-4)

    def test_module_from_every_seed(self):
        proteins = ec.proteins()
        module = proteins.expand_seed_set(self.SEEDS, min_score=0.5)
        added = [n for n in labels(proteins, module) if n not in self.SEEDS]
        assert sorted(added) == ["ORF1", "REP3", "REP4", "REP6"]
        hsp1 = proteins.get_neighbors("HSP1")
        assert len(hsp1) == 9 and len(set(hsp1) & set(module)) < 9 / 2
        grown = labels(proteins, proteins.expand_seed_set(self.SEEDS))
        assert len(grown) == 15 and "HSP1" not in grown
        assert {"CYC1", "CYC3", "CYC4", "CYC5", "TRN4", "ORF3"} <= set(grown)

    def test_steiner_tree_edges(self):
        proteins = ec.proteins()
        prizes = {s: 3.0 for s in self.SEEDS}
        assert proteins.prize_collecting_steiner_tree(prizes).count_edges() == 4
        cheaper = proteins.prize_collecting_steiner_tree(prizes, edge_penalty=0.5)
        assert sorted(cheaper.get_node_df()["label"]) == [
            "CYC2",
            "HSP1",
            "REP1",
            "REP2",
            "REP3",
            "REP5",
            "TRN2",
            "TRN3",
        ]

    def test_smoothing_moves_rep3_and_enz3_most(self):
        proteins = ec.proteins()
        expression = proteins.get_node_attrs("expression")
        smoothed = proteins.network_smooth(expression, return_dict=True)
        moved = {proteins.get_label_from_id(n): smoothed[n] - expression[n] for n in expression}
        assert max(moved, key=moved.get) == "REP3"
        assert min(moved, key=moved.get) == "ENZ3"

    def test_module_pathway_enrichment(self):
        proteins = ec.proteins()
        module = proteins.expand_seed_set(self.SEEDS[:4], min_score=0.5)
        cell_cycle = proteins.subgraph_enrichment(module, "pathway", "Cell cycle")
        assert (cell_cycle["k"], cell_cycle["K"]) == (1, 5)
        assert cell_cycle["fold_enrichment"] == pytest.approx(0.775)
        communities = proteins.cmty_louvain(return_dict=True)
        assert len(set(communities.values())) == 5
        results = proteins.module_annotation_enrichment(communities, "pathway")
        assert sum(len(by_value) for by_value in results.values()) == 30
        best = {
            value: (r["k"], r["n"], r["p_adjusted"])
            for by_value in results.values()
            for value, r in by_value.items()
            if r["fold_enrichment"] > 1 and r["k"] > 1
        }
        assert set(best) == {"DNA repair", "Cell cycle", "Signaling", "Transport", "Metabolism"}
        assert best["Transport"][:2] == (4, 6)
        assert sorted(v for v in best if best[v][2] < 0.05) == [
            "Cell cycle",
            "DNA repair",
            "Metabolism",
            "Signaling",
        ]
        assert best["Transport"][2] == pytest.approx(0.0673, abs=5e-5)
        transport = [
            proteins.get_label_from_id(n)
            for n, c in communities.items()
            if c == communities[proteins.get_id_from_label("TRN2")]
        ]
        assert sorted(transport) == ["HSP1", "ORF3", "TRN2", "TRN3", "TRN4", "TRN5"]

    def test_gwas_pipeline(self):
        proteins = ec.proteins()
        result = proteins.gwas_pipeline(
            proteins.get_node_attrs("expression"), enrichment_attr="disease_gene"
        )
        ranked = list(result["module_scores"])
        top = labels(proteins, result["communities"][ranked[0]])
        assert sorted(top) == ["ORF1", "REP1", "REP2", "REP3", "REP4", "REP5", "REP6"]
        assert result["module_scores"][ranked[0]] == pytest.approx(1.631, abs=5e-4)
        assert result["enrichment"][ranked[0]] == pytest.approx(3 / (5 * 7 / 31))
        last = labels(proteins, result["communities"][ranked[-1]])
        assert "SIG2" in last and not set(last) & set(self.SEEDS)
        assert result["module_scores"][ranked[-1]] == pytest.approx(-0.012, abs=5e-4)


class TestFoodWeb:
    def test_trophic_levels(self):
        web = ec.food_web()
        levels = {
            web.get_label_from_id(n): v for n, v in web.trophic_levels(return_dict=True).items()
        }
        kinds = dict(zip(web.get_node_df()["label"], web.get_node_df()["kind"]))
        assert all(levels[s] == 1 for s in kinds if kinds[s] == "plant")
        assert all(levels[s] == 2 for s in kinds if kinds[s] == "herbivore")
        assert levels["Jay"] == pytest.approx(2.5)
        assert levels["Fox"] == pytest.approx(11 / 3)
        assert levels["Hawk"] == pytest.approx(3.5)
        assert web.trophic_incoherence_parameter() == pytest.approx(0.28, abs=0.005)

    def test_drawn_by_trophic_level_without_crossings(self):
        web = ec.food_web()
        pos = {n: tuple(p) for n, p in web.get_node_attrs("pos").items()}
        levels = web.trophic_levels(return_dict=True)
        assert all(pos[n][1] == pytest.approx(2 * levels[n], abs=0.001) for n in pos)
        assert_crossing_free(web, pos)

    def test_top_predators(self):
        web = ec.food_web()
        tops = sorted(labels(web, c) for c in web.attracting_components())
        assert tops == [["Fox"], ["Hawk"], ["Marten"]]

    def test_generations_differ_from_trophic_levels(self):
        web = ec.food_web()
        generations = [labels(web, g) for g in web.topological_generations()]
        assert generations == [
            ["Bramble", "Clover", "Grass", "Oak"],
            ["Caterpillar", "Grasshopper", "Rabbit", "Squirrel", "Vole"],
            ["Frog", "Jay", "Marten", "Shrew", "Weasel"],
            ["Fox", "Hawk"],
        ]
        levels = web.trophic_levels(return_dict=True)
        assert levels[web.get_id_from_label("Jay")] == pytest.approx(2.5)

    def test_what_each_predator_lives_on(self):
        web = ec.food_web()
        hawk, fox = web.ancestors("Hawk"), web.ancestors("Fox")
        assert (len(hawk), len(fox)) == (8, 7)
        assert labels(web, set(hawk) & set(fox)) == ["Bramble", "Grass", "Rabbit"]
        oak = web.get_id_from_label("Oak")
        assert oak in hawk and oak not in fox

    def test_keystone_plant(self):
        web = ec.food_web()
        kinds = dict(zip(web.get_node_df()["label"], web.get_node_df()["kind"]))
        food = {s: set(labels(web, web.get_predecessors(s))) for s in kinds}

        def starved(lost):
            alive = set(kinds) - {lost}
            while True:
                gone = {s for s in alive if kinds[s] != "plant" and not food[s] & alive}
                if not gone:
                    return sorted(set(kinds) - alive - {lost})
                alive -= gone

        assert starved("Oak") == ["Caterpillar", "Jay", "Marten", "Squirrel"]
        assert all(starved(p) == [] for p in ["Clover", "Grass", "Bramble"])

    def test_plants_that_feed_the_most(self):
        web = ec.food_web()
        reach = web.local_reaching_centrality(return_dict=True)
        best = max(reach.values())
        assert best == pytest.approx(7 / 15)
        assert labels(web, [n for n, v in reach.items() if v == best]) == ["Grass", "Oak"]


class TestControlFlow:
    def test_one_entry_one_exit(self):
        cfg = ec.control_flow()
        sources = [n for n in cfg.get_node_ids() if not cfg.get_predecessors(n)]
        sinks = [n for n in cfg.get_node_ids() if not cfg.get_successors(n)]
        assert labels(cfg, sources) == ["entry"] and labels(cfg, sinks) == ["exit"]
        branches = cfg.get_edge_df()["branch"].drop_nulls().to_list()
        assert sorted(branches) == [
            "done",
            "false",
            "false",
            "false",
            "next",
            "true",
            "true",
            "true",
        ]

    def test_drawn_without_crossings(self):
        cfg = ec.control_flow()
        assert_crossing_free(cfg, {n: tuple(p) for n, p in cfg.get_node_attrs("pos").items()})

    def test_one_loop(self):
        cfg = ec.control_flow()
        loops = [labels(cfg, c) for c in cfg.strongly_connected_components() if len(c) > 1]
        assert loops == [["add", "limit test", "loop", "negate", "sign test"]]

    def test_dominators(self):
        cfg = ec.control_flow()
        label = cfg.get_label_from_id
        idom = {label(k): label(v) for k, v in cfg.immediate_dominators("entry").items()}
        assert idom["add"] == "sign test"
        assert idom["exit"] == "empty test"
        assert idom["empty test"] == "loop"
        assert idom["loop"] == "entry"
        assert "negate" not in idom.values()

    def test_dominance_frontiers(self):
        cfg = ec.control_flow()
        frontiers = {
            cfg.get_label_from_id(k): labels(cfg, v)
            for k, v in cfg.dominance_frontiers("entry").items()
        }
        assert frontiers["negate"] == ["add"]
        assert frontiers["loop"] == ["loop"]
        assert frontiers["break"] == ["empty test"]
