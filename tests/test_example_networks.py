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
        assert ec.payments().reciprocity() == pytest.approx(4 / 35)

    def test_strong_components(self):
        payments = ec.payments()
        big = [labels(payments, c) for c in payments.strongly_connected_components() if len(c) > 1]
        assert sorted(big) == [
            ["Ana", "Ben", "Cruz", "Dee", "Eve", "Finn", "Gus", "Hal"],
            ["Max", "Nia"],
        ]
        assert payments.is_weakly_connected()


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
        assert sorted(scores.values())[-2] < scores[best]

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
        # greedy doubles back
        assert len(towns.traveling_salesman(method="greedy")) > 13

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
        assert not benzene.is_isomorphic(cyclohexane, edge_attr="bond")

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
