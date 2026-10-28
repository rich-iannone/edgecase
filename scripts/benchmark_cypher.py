#!/usr/bin/env python3
"""Benchmark the Cypher matcher: the Rust engine against the Python reference, and optionally
against Neo4j.

Two random directed graphs are built with a fixed seed: a social graph (people who know each
other and work at companies) and a movie graph (people who act in and direct movies). Each
query is timed as `Matcher.count()` with both engines and as `Graph.match()` (the DataFrame,
with the Rust engine), best of `--repeat` runs. The match counts of the engines must agree.

With `--neo4j URI`, the same graphs are loaded into Neo4j (with the `neo4j` driver, which must
be installed) and each query is timed there too, as `<query> RETURN count(*)`, after one warm-up
run. The loaded nodes carry the extra label `EdgecaseBench`, and are deleted before and after.
The database must hold no other nodes, since the queries would match them too.

Usage:
    python scripts/benchmark_cypher.py [--nodes 10000] [--edges 50000] [--repeat 3]
                                       [--seed 1] [--no-python]
                                       [--neo4j bolt://localhost:7687 --user neo4j
                                        --password secret]
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Any, Optional

import edgecase as ec
from edgecase.cypher import Matcher, parse_query

MARKER = "EdgecaseBench"


# ---------------------------------------------------------------------------
# Graphs: (nodes, edges) as plain data, so EdgeCase and Neo4j get the same graph
# ---------------------------------------------------------------------------
# nodes: [(key, type, properties)]; edges: [(source key, target key, rel, properties)]


def social_graph(n: int, m: int, seed: int):
    rng = random.Random(seed)
    companies = max(1, n // 20)
    nodes = [(f"c{i}", "Company", {"name": f"c{i}", "founded": rng.randint(1900, 2020)})
             for i in range(companies)]
    people = n - companies
    nodes += [(f"p{i}", "Person", {"name": f"p{i}", "age": rng.randint(18, 90)})
              for i in range(people)]
    edges = []
    for i in range(people):
        edges.append((f"p{i}", f"c{rng.randrange(companies)}", "WORKS_AT",
                      {"since": rng.randint(1990, 2024)}))
    while len(edges) < m:
        a, b = rng.randrange(people), rng.randrange(people)
        if a != b:
            edges.append((f"p{a}", f"p{b}", "KNOWS", {"since": rng.randint(1990, 2024)}))
    return nodes, edges


def movie_graph(n: int, m: int, seed: int):
    rng = random.Random(seed)
    movies = max(1, n // 5)
    people = n - movies
    nodes = [(f"m{i}", "Movie", {"title": f"m{i}", "released": rng.randint(1950, 2024)})
             for i in range(movies)]
    nodes += [(f"p{i}", "Person", {"name": f"p{i}", "born": rng.randint(1920, 2005)})
              for i in range(people)]
    edges = [(f"p{rng.randrange(people)}", f"m{i}", "DIRECTED", {}) for i in range(movies)]
    # popular people act more often, as in real casts
    weights = [1 / (i + 1) ** 0.5 for i in range(people)]
    actors = rng.choices(range(people), weights=weights, k=max(0, m - len(edges)))
    for actor in actors:
        edges.append((f"p{actor}", f"m{rng.randrange(movies)}", "ACTED_IN",
                      {"year": rng.randint(1950, 2024)}))
    return nodes, edges


def to_edgecase(nodes, edges) -> ec.Graph:
    g = ec.Graph(directed=True)
    g.add_nodes([(key, {"type": t, **props}) for key, t, props in nodes])
    g.add_edges([(s, t, {"rel": rel, **props}) for s, t, rel, props in edges])
    return g


# ---------------------------------------------------------------------------
# Queries: (name, graph, query, params); no RETURN, so Neo4j can append one
# ---------------------------------------------------------------------------

QUERIES = [
    ("one hop", "social", "MATCH (a:Person)-[:KNOWS]->(b:Person)", None),
    ("two hops, filtered", "social",
     "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) WHERE a.age < 30 AND c.age > 80", None),
    ("anchored three hops", "social",
     "MATCH (a:Person {name: $name})-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(d)",
     {"name": "p7"}),
    ("triangles", "social", "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a)", None),
    ("variable length *1..3", "social",
     "MATCH (a:Person {name: $name})-[:KNOWS*1..3]->(b)", {"name": "p7"}),
    ("colleagues", "social",
     "MATCH (a:Person)-[:WORKS_AT]->(c:Company)<-[:WORKS_AT]-(b:Person) "
     "WHERE a.age > 85 AND b.age > 85", None),
    ("optional match", "social",
     "MATCH (p:Person) WHERE p.age > 85 OPTIONAL MATCH (p)-[:KNOWS]->(q:Person) "
     "WHERE q.age < 20", None),
    ("Python condition", "social",
     "MATCH (a:Person)-[:KNOWS]->(b) WHERE a.name STARTS WITH 'p1'", None),
    ("co-actors", "movies",
     "MATCH (a:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(b:Person) WHERE id(a) < id(b)",
     None),
    ("directors who act", "movies",
     "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(n:Movie) "
     "WHERE m.released > n.released", None),
]


def best(fn, repeat: int) -> tuple[float, Any]:
    """The shortest time of `repeat` calls of `fn`, and its result."""
    times, result = [], None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return min(times), result


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------


class Neo4j:
    def __init__(self, uri: str, user: str, password: str):
        try:
            import neo4j
        except ImportError:
            raise SystemExit("--neo4j needs the Neo4j driver: pip install neo4j") from None
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.clear()
        others = self.run(f"MATCH (n) WHERE NOT n:{MARKER} RETURN count(n) AS n")
        if others:
            raise SystemExit(f"the Neo4j database holds {others} other nodes, which the "
                             "queries would match too; use an empty database")
        self.run(f"CREATE INDEX edgecase_bench IF NOT EXISTS FOR (n:{MARKER}) ON (n.key)")

    def run(self, query: str, params: Optional[dict] = None) -> Any:
        with self.driver.session() as session:
            record = session.run(query, params or {}).single()
            return None if record is None else record[0]

    def clear(self):
        while self.run(f"MATCH (n:{MARKER}) WITH n LIMIT 10000 DETACH DELETE n "
                       "RETURN count(*) AS n"):
            pass

    def load(self, nodes, edges, batch: int = 5000):
        self.clear()
        by_type: dict[str, list] = {}
        for key, t, props in nodes:
            by_type.setdefault(t, []).append({"key": key, **props})
        for t, rows in by_type.items():
            for i in range(0, len(rows), batch):
                self.run(f"UNWIND $rows AS row CREATE (n:{MARKER}:{t}) SET n = row",
                         {"rows": rows[i:i + batch]})
        by_rel: dict[str, list] = {}
        for s, t, rel, props in edges:
            by_rel.setdefault(rel, []).append({"s": s, "t": t, "props": props})
        for rel, rows in by_rel.items():
            for i in range(0, len(rows), batch):
                self.run(f"UNWIND $rows AS row MATCH (a:{MARKER} {{key: row.s}}), "
                         f"(b:{MARKER} {{key: row.t}}) CREATE (a)-[r:{rel}]->(b) "
                         "SET r = row.props", {"rows": rows[i:i + batch]})

    def count(self, query: str, params: Optional[dict]) -> int:
        return self.run(f"{query} RETURN count(*) AS n", params)


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--edges", type=int, default=50_000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-python", action="store_true",
                        help="skip the Python engine, which is slow on big graphs")
    parser.add_argument("--neo4j", metavar="URI", help="also time the queries in Neo4j")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument("--password", default="neo4j")
    args = parser.parse_args()

    data = {
        "social": social_graph(args.nodes, args.edges, args.seed),
        "movies": movie_graph(args.nodes, args.edges, args.seed),
    }
    graphs = {name: to_edgecase(*d) for name, d in data.items()}
    for name, g in graphs.items():
        print(f"{name}: {g.count_nodes():,} nodes, {g.count_edges():,} edges")
    neo = Neo4j(args.neo4j, args.user, args.password) if args.neo4j else None
    print()

    header = f"{'query':<24}{'matches':>10}{'rust':>10}{'python':>10}{'speedup':>9}" \
             f"{'match()':>10}" + (f"{'neo4j':>10}" if neo else "")
    print(header)
    print("-" * len(header))
    loaded = None
    for name, graph_name, text, params in QUERIES:
        g = graphs[graph_name]
        query = parse_query(text)

        def count(engine, g=g, query=query, text=text, params=params):
            return Matcher(query, text, g._graph, params, engine=engine).count()[0]

        rust_time, n = best(lambda: count("rust"), args.repeat)
        cells = [f"{name:<24}{n:>10,}{rust_time:>10.4f}"]
        if args.no_python:
            cells.append(f"{'':>10}{'':>9}")
        else:
            python_time, python_n = best(lambda: count("python"), args.repeat)
            if python_n != n:
                raise SystemExit(f"{name}: the engines disagree, {n} and {python_n} matches")
            cells.append(f"{python_time:>10.4f}{python_time / rust_time:>8.1f}x")
        match_time, df = best(lambda g=g: g.match(text, params=params), args.repeat)
        cells.append(f"{match_time:>10.4f}")
        if neo:
            if loaded != graph_name:
                neo.load(*data[graph_name])
                loaded = graph_name
            neo.count(text, params)  # warm-up: query planning and caches
            neo_time, neo_n = best(lambda: neo.count(text, params), args.repeat)
            cells.append(f"{neo_time:>10.4f}" + ("" if neo_n == n else f"  ({neo_n:,} matches)"))
        print("".join(cells), flush=True)
    if neo:
        neo.clear()
        neo.driver.close()
    print("\ntimes in seconds, best of", args.repeat)


if __name__ == "__main__":
    main()
