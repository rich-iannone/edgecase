"""
Small example networks for trying out methods and for the examples in the docs.

Each network is a few dozen nodes or fewer, with human-readable labels and attributes in real
units, and is designed so the methods it serves have something to find: the metro has a river
crossing that is a bridge, the pipeline a bottleneck that limits its flow, the exam timetable a
greedy order that wastes a slot. Those planted features are listed in each loader's docstring and
checked by the test suite, so the docs can rely on them. Unlike the datasets, the networks are
fictional, and make no claims about a real city, company or school. The two exceptions are
textbook cases: the bridges of Königsberg and a handful of well-known molecules.
"""

from __future__ import annotations

import math

from edgecase.graph import Graph
from edgecase.io import _build_graph

__all__ = [
    "causal",
    "citations",
    "control_flow",
    "courses",
    "currency",
    "exams",
    "food_web",
    "friends",
    "konigsberg",
    "metro",
    "molecules",
    "offices",
    "org_chart",
    "payments",
    "pipeline",
    "project",
    "proteins",
    "teams",
    "towns",
]


# Stations as name: (x, y, zone), with x and y in kilometers from Central
_METRO_STATIONS = {
    "Westgate": (-6.0, 0.0, 3),
    "Mill Lane": (-4.0, 0.0, 2),
    "Old Town": (-2.0, 0.0, 1),
    "Central": (0.0, 0.0, 1),
    "Market": (2.0, 0.0, 1),
    "Eastfield": (4.0, 0.0, 2),
    "Lakeside": (6.5, 0.5, 3),
    "Northpark": (0.0, 6.5, 3),
    "University": (0.0, 4.0, 2),
    "Museum": (0.0, 2.0, 1),
    "Riverside": (0.0, -2.0, 1),
    "Harbor": (0.0, -4.5, 2),
    "Docks": (-1.8, -6.0, 3),
    "Shipyard": (1.8, -6.0, 3),
    "Hillcrest": (-4.5, 4.5, 3),
    "Garden": (4.5, 4.5, 3),
    "Cathedral": (-1.5, 1.5, 1),
    "Opera": (1.5, 1.5, 1),
    "Business Park": (6.5, -2.5, 3),
    "Airport": (8.5, -4.5, 3),
}

# Track segments as (line, station, station, minutes)
_METRO_SEGMENTS = [
    ("Red", "Westgate", "Mill Lane", 4),
    ("Red", "Mill Lane", "Old Town", 3),
    ("Red", "Old Town", "Central", 2),
    ("Red", "Central", "Market", 2),
    ("Red", "Market", "Eastfield", 3),
    ("Red", "Eastfield", "Lakeside", 4),
    ("Blue", "Northpark", "University", 3),
    ("Blue", "University", "Museum", 3),
    ("Blue", "Museum", "Central", 2),
    ("Blue", "Central", "Riverside", 2),
    ("Blue", "Riverside", "Harbor", 3),
    ("Blue", "Harbor", "Docks", 3),
    ("Blue", "Harbor", "Shipyard", 3),
    ("Green", "Westgate", "Hillcrest", 5),
    ("Green", "Hillcrest", "Northpark", 6),
    ("Green", "Northpark", "Garden", 6),
    ("Green", "Garden", "Lakeside", 5),
    ("Ring", "Old Town", "Cathedral", 2),
    ("Ring", "Cathedral", "Museum", 2),
    ("Ring", "Museum", "Opera", 2),
    ("Ring", "Opera", "Market", 2),
    ("Ring", "Market", "Riverside", 3),
    ("Ring", "Riverside", "Old Town", 3),
    ("Airport", "Eastfield", "Business Park", 4),
    ("Airport", "Business Park", "Airport", 5),
]


def metro() -> Graph:
    """
    A city's metro network, with travel times.

    The metro of a made-up city: 20 stations on a Red line running west to east, a Blue line
    running north to south across the river, a Green line arcing around the northern suburbs, a
    Ring line around downtown, and an Airport line. Each edge is a stretch of track between two
    stations, weighted by the minutes it takes, so weighted methods measure routes in travel
    time. Every station has map coordinates, so `plot(layout="pos")` draws it as a map.

    Network Properties:

    - 20 nodes (stations, labeled by name)
    - 25 edges (track between neighboring stations)
    - Undirected, weighted by travel time in minutes
    - Connected

    Node Attributes:

    - `lines`: the lines that stop at the station, such as `["Red", "Blue"]`
    - `zone`: the fare zone, from `1` (downtown) to `3` (the outer suburbs)
    - `pos`: the station's `(x, y)` position on the map, in kilometers from Central

    Edge Attributes:

    - `line`: the line the track belongs to (`"Red"`, `"Blue"`, `"Green"`, `"Ring"` or
      `"Airport"`)
    - `minutes`: the travel time, the same as the edge weight

    What to Look For:

    - **One river crossing.** The Blue line's Riverside–Harbor stretch is the only way across
      the river, so it's a bridge: closing it cuts off Harbor, Docks and Shipyard. The other
      bridges are the two branches beyond Harbor and the two stretches of the Airport line.
    - **Single points of failure.** The articulation points are Riverside, Harbor, Eastfield
      (where the Airport line branches off) and Business Park. Central isn't one: the Ring line
      keeps downtown connected without it.
    - **Two kinds of central.** Central has the highest closeness (the shortest average trip to
      everywhere else), but Market has the highest betweenness, because every trip between the
      east side and the rest of the city passes through it.
    - **Fastest isn't fewest stops.** From Westgate to Lakeside, the Green line takes 4 stops
      but 22 minutes; the Red line through downtown takes 6 stops and only 18.
    - **A* has a heuristic.** No train goes faster than 1 km a minute, so the straight-line
      distance between two stations, in kilometers, never overestimates the minutes left.

    Returns
    -------
    Graph
        The metro network.

    Examples
    --------
    Draw the metro as a map, with the stations colored by fare zone:

    ```{python}
    import edgecase as ec

    metro = ec.metro()

    metro.plot(layout="pos", color_by="zone", label_by="label")
    ```

    The fastest route from Westgate to Lakeside goes through downtown, while the route with the
    fewest stops takes the Green line around it:

    ```{python}
    fastest = metro.shortest_path("Westgate", "Lakeside", return_list=True)
    fewest = metro.astar_path("Westgate", "Lakeside", weighted=False)

    [metro.get_label_from_id(i) for i in fastest], [metro.get_label_from_id(i) for i in fewest]
    ```

    The bridges are the stretches of track the network can't lose without cutting stations off:

    ```{python}
    metro.plot(layout="pos", highlight_edges=metro.bridges(), label_by="label")
    ```
    """
    lines: dict[str, list[str]] = {station: [] for station in _METRO_STATIONS}
    for line, u, v, _ in _METRO_SEGMENTS:
        for station in (u, v):
            if line not in lines[station]:
                lines[station].append(line)
    nodes = [
        (name, name, {"lines": lines[name], "zone": zone, "pos": (x, y)})
        for name, (x, y, zone) in _METRO_STATIONS.items()
    ]
    edges = [
        (u, v, float(minutes), {"line": line, "minutes": minutes})
        for line, u, v, minutes in _METRO_SEGMENTS
    ]
    return _build_graph(nodes, edges, False, True, "Metro", "metro()")


_PAYMENT_ACCOUNTS = {
    "Ana": "person",
    "Ben": "person",
    "Cruz": "person",
    "Dee": "person",
    "Eve": "person",
    "Finn": "person",
    "Gus": "person",
    "Hal": "person",
    "Ivy": "person",
    "Jo": "person",
    "Kai": "person",
    "Lena": "person",
    "Max": "person",
    "Nia": "person",
    "Grocer": "business",
    "Cafe": "business",
    "Landlord": "business",
    "Acme": "business",
}

# Payments as (payer, payee, amount, date)
_PAYMENTS = [
    ("Acme", "Dee", 2600, "2026-03-01"),
    ("Acme", "Max", 2400, "2026-03-01"),
    ("Acme", "Nia", 2500, "2026-03-01"),
    ("Acme", "Eve", 2700, "2026-03-01"),
    ("Dee", "Eve", 80, "2026-03-02"),
    ("Max", "Landlord", 1800, "2026-03-02"),
    ("Nia", "Max", 900, "2026-03-02"),
    ("Eve", "Dee", 25, "2026-03-03"),
    ("Dee", "Grocer", 64, "2026-03-03"),
    ("Ana", "Dee", 120, "2026-03-04"),
    ("Eve", "Grocer", 38, "2026-03-04"),
    ("Dee", "Cafe", 12, "2026-03-04"),
    ("Dee", "Ben", 60, "2026-03-05"),
    ("Max", "Grocer", 82, "2026-03-05"),
    ("Nia", "Cafe", 9, "2026-03-05"),
    ("Ben", "Eve", 300, "2026-03-06"),
    ("Nia", "Grocer", 57, "2026-03-06"),
    ("Gus", "Dee", 45, "2026-03-07"),
    ("Hal", "Grocer", 23, "2026-03-07"),
    ("Max", "Nia", 40, "2026-03-08"),
    ("Lena", "Grocer", 41, "2026-03-08"),
    ("Eve", "Finn", 2000, "2026-03-09"),
    ("Ana", "Grocer", 95, "2026-03-09"),
    ("Ana", "Ben", 9500, "2026-03-10"),
    ("Gus", "Grocer", 30, "2026-03-10"),
    ("Ben", "Cruz", 9400, "2026-03-11"),
    ("Eve", "Cafe", 15, "2026-03-11"),
    ("Cruz", "Ana", 9300, "2026-03-12"),
    ("Cruz", "Ivy", 3000, "2026-03-13"),
    ("Ivy", "Jo", 2950, "2026-03-14"),
    ("Jo", "Kai", 2900, "2026-03-15"),
    ("Kai", "Lena", 2850, "2026-03-16"),
    ("Finn", "Gus", 4950, "2026-03-16"),
    ("Gus", "Hal", 4900, "2026-03-17"),
    ("Hal", "Finn", 4850, "2026-03-18"),
]


def payments() -> Graph:
    """
    A month of payments between bank accounts.

    The payments made in one month between 18 accounts at a made-up bank, for fraud-detection
    and money-flow examples: an employer paying wages, people paying rent and buying groceries,
    and, hidden among the everyday spending, two rings of large payments and a chain of
    accounts passing money along. Each edge is one payment, directed from payer to payee and
    weighted by its amount. It extends the network on the EdgeCase homepage, whose 13 payments
    are all here.

    Network Properties:

    - 18 nodes (accounts, labeled by name)
    - 35 edges (payments)
    - Directed, weighted by the amount paid
    - Weakly connected

    Node Attributes:

    - `kind`: `"person"` or `"business"` (the employer Acme, the Grocer, the Cafe and the
      Landlord)

    Edge Attributes:

    - `amount`: the amount paid, the same as the edge weight
    - `date`: the day of the payment, as an ISO date string such as `"2026-03-10"`

    What to Look For:

    - **Two rings.** Ana, Ben and Cruz pass about 9,400 around a circle on three consecutive
      days, and Finn, Gus and Hal about 4,900. Each payment in a ring is 1,000 or more.
    - **A pass-through chain.** The day after the first ring closes, Cruz sends 3,000 to Ivy,
      who passes it to Jo, who passes it to Kai, who passes it to Lena, a day apart and 50
      less each time. Ivy, Jo and Kai each have exactly one payment in and one out.
    - **A hub.** The Grocer is paid by 8 accounts, more than any other, and pays no one.
    - **Mutual pairs.** Dee and Eve pay each other, and so do the flatmates Max and Nia: 4
      of the 35 payments are reciprocated.
    - **Strong components.** The six accounts in the rings, together with Dee and Eve, can
      all reach each other, and so can Max and Nia; every other account is on its own.

    Returns
    -------
    Graph
        The payments network.

    Examples
    --------
    Draw the payments, with businesses and people in different colors:

    ```{python}
    import edgecase as ec

    payments = ec.payments()

    payments.plot(color_by="kind", label_by="label", seed=3)
    ```

    Cypher finds the rings: three accounts that pay each other in a circle, 1,000 or more each
    time:

    ```{python}
    payments.match('''
        MATCH (a)-[x]->(b)-[y]->(c)-[z]->(a)
        WHERE x.amount >= 1000 AND y.amount >= 1000 AND z.amount >= 1000
          AND a.name < b.name AND a.name < c.name
        RETURN a.name AS a, b.name AS b, c.name AS c, x.date AS starts
    ''')
    ```

    Money passing through an account looks like a large payment in followed by a slightly
    smaller one out. Cypher finds the pairs of payments, and Polars keeps those where at most
    100 stays behind. It catches the chain from Ivy to Kai, and the ring members too:

    ```{python}
    import polars as pl

    (
        payments.match('''
            MATCH (a)-[x]->(b)-[y]->(c)
            WHERE x.amount >= 1000 AND y.amount < x.amount AND a <> c
            RETURN b.name AS account, x.amount AS received, y.amount AS sent, y.date AS sent_on
        ''')
        .filter(pl.col("received") - pl.col("sent") <= 100)
        .sort("sent_on")
    )
    ```
    """
    nodes = [(name, name, {"kind": kind}) for name, kind in _PAYMENT_ACCOUNTS.items()]
    edges = [
        (payer, payee, float(amount), {"amount": amount, "date": date})
        for payer, payee, amount, date in _PAYMENTS
    ]
    return _build_graph(nodes, edges, True, True, "Payments", "payments()")


# Courses as code: (title, department, credits)
_COURSES = {
    "CS101": ("Introduction to Programming", "CS", 4),
    "CS102": ("Data Structures", "CS", 4),
    "CS201": ("Algorithms", "CS", 4),
    "CS210": ("Computer Architecture", "CS", 3),
    "CS220": ("Software Engineering", "CS", 3),
    "CS250": ("Databases", "CS", 3),
    "CS310": ("Operating Systems", "CS", 4),
    "CS320": ("Computer Networks", "CS", 3),
    "CS350": ("Data Mining", "CS", 3),
    "CS360": ("Machine Learning", "CS", 4),
    "CS410": ("Distributed Systems", "CS", 3),
    "CS460": ("Deep Learning", "CS", 3),
    "MATH101": ("Calculus I", "MATH", 4),
    "MATH102": ("Calculus II", "MATH", 4),
    "MATH150": ("Discrete Mathematics", "MATH", 3),
    "MATH210": ("Linear Algebra", "MATH", 3),
    "STAT201": ("Probability and Statistics", "STAT", 3),
}

# Prerequisites as (prerequisite, course)
_PREREQUISITES = [
    ("CS101", "CS102"),
    ("CS101", "CS201"),
    ("CS102", "CS201"),
    ("MATH150", "CS201"),
    ("CS102", "CS210"),
    ("CS102", "CS220"),
    ("CS102", "CS250"),
    ("CS210", "CS310"),
    ("CS210", "CS320"),
    ("CS310", "CS410"),
    ("CS320", "CS410"),
    ("CS250", "CS350"),
    ("STAT201", "CS350"),
    ("CS201", "CS360"),
    ("STAT201", "CS360"),
    ("CS360", "CS460"),
    ("MATH101", "MATH102"),
    ("MATH102", "MATH210"),
    ("MATH210", "STAT201"),
]


def courses() -> Graph:
    """
    A degree program's courses and their prerequisites.

    The courses of a made-up computer science program, with an edge from each course to every
    course that requires it, so the graph is a directed acyclic graph (DAG) and a topological
    order is an order in which a student could take them. It's the example for DAG methods:
    scheduling, ancestors and descendants, longest paths and transitive reduction.

    Network Properties:

    - 17 nodes (courses, labeled by course code)
    - 19 edges (from a prerequisite to the course that requires it)
    - Directed, acyclic, unweighted

    Node Attributes:

    - `title`: the course's name, such as `"Data Structures"`
    - `dept`: the department, `"CS"`, `"MATH"` or `"STAT"`
    - `credits`: the credit hours, `3` or `4`
    - `level`: the course level from its code, `100` to `400`

    What to Look For:

    - **A course everything depends on.** All 11 other CS courses require CS101, directly or
      through other courses.
    - **The math sequence sets the pace.** The longest chain of prerequisites is MATH101,
      MATH102, MATH210, STAT201, CS360, CS460, so the program takes at least 6 semesters
      however the courses are scheduled; no chain of CS courses alone is longer than 5.
    - **A redundant prerequisite.** CS201 lists CS101 as a prerequisite, but it also requires
      CS102, which requires CS101, so the transitive reduction drops that one edge.
    - **Two independent tracks.** The systems courses (CS210, CS310, CS320, CS410) and the data
      courses (CS250, CS350, CS360, CS460) don't depend on each other, and meet only in CS102,
      the lowest common ancestor of CS410 and CS460.

    Returns
    -------
    Graph
        The prerequisite DAG.

    Examples
    --------
    Lay the courses out by level, colored by department:

    ```{python}
    import edgecase as ec

    courses = ec.courses()

    courses.plot(color_by="dept", label_by="label", seed=2)
    ```

    The topological generations are the fastest schedule: each semester, take every course
    whose prerequisites are done:

    ```{python}
    [[courses.get_label_from_id(i) for i in semester] for semester in courses.topological_generations()]
    ```
    """
    nodes = [
        (
            code,
            code,
            {
                "title": title,
                "dept": dept,
                "credits": credits,
                "level": int(code.lstrip("ACHMST")) // 100 * 100,
            },
        )
        for code, (title, dept, credits) in _COURSES.items()
    ]
    edges = [(u, v, None, {}) for u, v in _PREREQUISITES]
    return _build_graph(nodes, edges, True, False, "Courses", "courses()")


# Facilities as name: (kind, x, y), drawn left to right from the source to the sink
_PIPELINE_SITES = {
    "Reservoir": ("source", 0.0, 0.0),
    "North Intake": ("intake", 1.0, 1.0),
    "South Intake": ("intake", 1.0, -1.0),
    "Hill Pump": ("pump", 2.0, -1.8),
    "West Plant": ("plant", 3.0, 1.0),
    "East Plant": ("plant", 3.0, -1.0),
    "Valley Junction": ("junction", 4.0, 1.0),
    "Ridge Tank": ("tank", 4.0, -1.0),
    "North District": ("district", 5.0, 1.0),
    "South District": ("district", 5.0, -1.0),
    "City": ("sink", 6.0, 0.0),
}

# Pipes as (from, to, capacity in megaliters per day, cost per megaliter)
_PIPES = [
    ("Reservoir", "North Intake", 60, 1),
    ("Reservoir", "South Intake", 50, 1),
    ("North Intake", "West Plant", 40, 2),
    ("South Intake", "East Plant", 10, 1),
    ("South Intake", "Hill Pump", 30, 3),
    ("Hill Pump", "East Plant", 30, 3),
    ("West Plant", "East Plant", 5, 1),
    ("West Plant", "Valley Junction", 25, 2),
    ("East Plant", "Ridge Tank", 20, 2),
    ("Valley Junction", "Ridge Tank", 10, 1),
    ("Valley Junction", "North District", 30, 1),
    ("Ridge Tank", "South District", 35, 1),
    ("North District", "City", 40, 0),
    ("South District", "City", 40, 0),
]


def pipeline() -> Graph:
    """
    A water supply network, with pipe capacities and costs.

    The water system of a made-up city: water flows from a reservoir through two intakes and
    two treatment plants, across a valley, and into the city's two districts. Each edge is a
    pipe with a capacity (the most water it can carry a day) and a cost per unit of water, so
    it's the example for flows and cuts: how much water can reach the city, which pipes limit
    it, and the cheapest way to deliver a given amount.

    Network Properties:

    - 11 nodes (facilities, labeled by name), with the source `"Reservoir"` and the sink
      `"City"`
    - 14 edges (pipes, in the direction the water flows)
    - Directed, unweighted (capacities and costs are attributes)

    Node Attributes:

    - `kind`: `"source"`, `"intake"`, `"pump"`, `"plant"`, `"junction"`, `"tank"`,
      `"district"` or `"sink"`
    - `pos`: an `(x, y)` position for a schematic drawing, from the reservoir on the left to the
      city on the right

    Edge Attributes:

    - `capacity`: the most the pipe can carry, in megaliters (ML) a day
    - `cost`: the cost of moving one megaliter through the pipe (pumping and treatment)

    The flow methods read the `capacity` attribute, and `minimum_cost_flow()` the `cost`
    attribute, without being told.

    What to Look For:

    - **A bottleneck across the valley.** At most 45 ML a day reaches the city, however the
      water is routed: the two pipes across the valley, West Plant to Valley Junction (25) and
      East Plant to Ridge Tank (20), form the minimum cut. Everything upstream could carry
      more.
    - **Cheap and narrow, or wide and costly.** The East Plant is fed by an old gravity
      aqueduct (10 ML a day at a cost of 1) and by the Hill Pump (30 ML a day at a cost of 3
      per pipe). The cheapest way to deliver 40 ML a day uses no pumping; only the last 5 of
      the 45 ML need the pump.
    - **Rising marginal cost.** Delivering 10 ML a day costs 50, 35 ML costs 200, 40 ML costs
      235 and 45 ML costs 285, as ever more expensive routes come into use.

    Returns
    -------
    Graph
        The pipeline network.

    Examples
    --------
    Draw the network as a schematic, from the reservoir to the city:

    ```{python}
    import edgecase as ec

    pipeline = ec.pipeline()

    pipeline.plot(layout="pos", color_by="kind", label_by="label", width=700, height=360)
    ```

    The most water the city can get, and the pipes that limit it:

    ```{python}
    value, (reservoir_side, city_side) = pipeline.minimum_cut("Reservoir", "City")

    value, [pipeline.get_label_from_id(i) for i in city_side]
    ```
    """
    nodes = [
        (name, name, {"kind": kind, "pos": (x, y)})
        for name, (kind, x, y) in _PIPELINE_SITES.items()
    ]
    edges = [(u, v, None, {"capacity": capacity, "cost": cost}) for u, v, capacity, cost in _PIPES]
    return _build_graph(nodes, edges, True, False, "Pipeline", "pipeline()")


# Courses with an exam this term, as name: students enrolled
_EXAM_COURSES = {
    "Calculus": 120,
    "Physics": 85,
    "Statistics": 70,
    "Chemistry": 95,
    "Biology": 110,
    "History": 60,
    "Philosophy": 45,
    "Literature": 50,
    "Psychology": 90,
    "Economics": 80,
    "Programming": 100,
    "Data Visualization": 40,
    "Databases": 55,
    "Web Design": 35,
}

# Pairs of courses with students in common, as (course, course, students in both)
_EXAM_CONFLICTS = [
    ("Calculus", "Physics", 12),
    ("Calculus", "Statistics", 6),
    ("Calculus", "Chemistry", 14),
    ("Calculus", "Biology", 22),
    ("Calculus", "Economics", 3),
    ("Calculus", "Programming", 4),
    ("Physics", "Chemistry", 19),
    ("Physics", "Biology", 5),
    ("Statistics", "Chemistry", 13),
    ("Statistics", "Data Visualization", 2),
    ("Statistics", "Databases", 10),
    ("Chemistry", "Biology", 8),
    ("Chemistry", "Economics", 3),
    ("Biology", "History", 3),
    ("Biology", "Philosophy", 8),
    ("Biology", "Psychology", 15),
    ("History", "Economics", 3),
    ("History", "Data Visualization", 5),
    ("Philosophy", "Literature", 3),
    ("Philosophy", "Psychology", 10),
    ("Literature", "Economics", 8),
    ("Psychology", "Economics", 3),
    ("Psychology", "Data Visualization", 3),
    ("Psychology", "Databases", 5),
    ("Economics", "Programming", 20),
    ("Programming", "Data Visualization", 2),
    ("Programming", "Databases", 11),
    ("Data Visualization", "Databases", 8),
    ("Data Visualization", "Web Design", 2),
]


def exams() -> Graph:
    """
    An exam timetabling problem, as a conflict graph.

    The 14 courses with a final exam this term at a made-up college, joined when some student
    takes both, so their exams can't be in the same time slot. Giving each course a slot so
    that no edge joins two courses in the same slot is graph coloring, and the fewest slots
    that work is the graph's chromatic number.

    Network Properties:

    - 14 nodes (courses, labeled by name)
    - 29 edges (pairs of courses with students in common)
    - Undirected, unweighted
    - Connected

    Node Attributes:

    - `students`: the number of students enrolled in the course

    Edge Attributes:

    - `shared`: the number of students taking both courses

    What to Look For:

    - **Four slots are needed.** Calculus, Physics, Chemistry and Biology all share students
      with each other, a clique of four, so no timetable has fewer than 4 slots. It's the only
      clique of four, and there is none of five.
    - **Four slots are enough.** The chromatic number is exactly 4: the DSATUR and
      smallest-last strategies of `greedy_color()` find a 4-slot timetable.
    - **A greedy order that wastes a slot.** The default `"largest_first"` strategy, which
      schedules the courses with the most conflicts first, needs 5 slots.

    Returns
    -------
    Graph
        The conflict graph.

    Examples
    --------
    Draw the conflicts, with each course sized by its enrollment:

    ```{python}
    import edgecase as ec

    exams = ec.exams()

    exams.plot(size_by="students", label_by="label", seed=5)
    ```

    Compare how many slots different greedy strategies need:

    ```{python}
    {
        strategy: len(set(exams.greedy_color(strategy=strategy).values()))
        for strategy in ["largest_first", "smallest_last", "dsatur"]
    }
    ```
    """
    nodes = [(name, name, {"students": students}) for name, students in _EXAM_COURSES.items()]
    edges = [(u, v, None, {"shared": shared}) for u, v, shared in _EXAM_CONFLICTS]
    return _build_graph(nodes, edges, False, False, "Exams", "exams()")


_TEAM_WORKERS = {
    "Priya": "Engineer",
    "Tom": "Engineer",
    "Mei": "Designer",
    "Omar": "Engineer",
    "Sara": "Writer",
    "Luis": "Designer",
    "Kofi": "Engineer",
    "Hana": "Tester",
}

_TEAM_TASKS = ["Backend", "Frontend", "Database", "Design", "Testing", "Docs", "DevOps", "Security"]

# Who can do what, as (worker, task, hours it would take them)
_TEAM_SKILLS = [
    ("Priya", "Backend", 6),
    ("Priya", "Frontend", 8),
    ("Priya", "Testing", 5),
    ("Tom", "Backend", 7),
    ("Tom", "Frontend", 5),
    ("Tom", "Testing", 6),
    ("Mei", "Design", 4),
    ("Mei", "Frontend", 7),
    ("Omar", "Database", 5),
    ("Omar", "DevOps", 6),
    ("Omar", "Security", 8),
    ("Omar", "Backend", 9),
    ("Sara", "Docs", 3),
    ("Sara", "Testing", 7),
    ("Luis", "Design", 5),
    ("Luis", "Docs", 6),
    ("Kofi", "Database", 6),
    ("Kofi", "Security", 5),
    ("Kofi", "DevOps", 7),
    ("Hana", "Testing", 4),
    ("Hana", "Docs", 5),
]

_TEAM_CONTRACTOR = [("Quinn", "Security", 4), ("Quinn", "DevOps", 5)]


def teams(contractor: bool = False) -> Graph:
    """
    Workers and the tasks they can do, as a bipartite graph.

    A made-up software team of 8 workers and the 8 tasks of a release, with an edge from each
    worker to every task they can do, weighted by the hours it would take them. Giving every
    task its own worker is a matching problem, and doing it in the fewest total hours is the
    assignment problem. The team is one person short, so set `contractor=True` to hire Quinn,
    a contractor who can take on security or DevOps.

    Network Properties:

    - 16 nodes (8 workers, then 8 tasks, labeled by name), or 17 with the contractor
    - 21 edges (who can do what), or 23 with the contractor
    - Undirected, bipartite, weighted by hours

    Node Attributes:

    - `side`: `"worker"` or `"task"`
    - `role`: the worker's job, `"Engineer"`, `"Designer"`, `"Writer"`, `"Tester"` or
      `"Contractor"` (missing for tasks)

    Edge Attributes:

    - `hours`: how long the worker would take on the task, the same as the edge weight

    What to Look For:

    - **One task too many.** Only Omar and Kofi can do any of Database, DevOps and Security,
      and two people can't take three tasks (Hall's condition fails), so at most 7 of the 8
      tasks can be covered and `bipartite_minimum_weight_full_matching()` raises an error.
    - **The contractor fixes it.** With `contractor=True` every task can be covered, and the
      cheapest assignment is unique: 37 hours, with Quinn on DevOps, Kofi on Security, Omar on
      Database, and Luis, one of the two designers, left free. The next best takes 38.

    Parameters
    ----------
    contractor
        Whether to add the contractor Quinn, who can do the `"Security"` task in 4 hours or the
        `"DevOps"` task in 5. Default is `False`.

    Returns
    -------
    Graph
        The bipartite graph of workers and tasks.

    Examples
    --------
    Draw the workers and tasks, colored by side:

    ```{python}
    import edgecase as ec

    team = ec.teams()

    team.plot(color_by="side", label_by="label", seed=4)
    ```

    Without the contractor, a maximum matching covers only 7 of the 8 tasks:

    ```{python}
    len(team.bipartite_maximum_matching())
    ```

    With the contractor, find the cheapest way to cover every task:

    ```{python}
    team = ec.teams(contractor=True)
    matched = team.bipartite_minimum_weight_full_matching()

    (
        team.layout_bipartite(align="horizontal", aspect_ratio=0.4)
        .plot(layout="pos", highlight_edges=matched, label_by="label", height=300)
    )
    ```
    """
    workers = dict(_TEAM_WORKERS)
    skills = list(_TEAM_SKILLS)
    if contractor:
        workers["Quinn"] = "Contractor"
        skills += _TEAM_CONTRACTOR
    nodes = [(name, name, {"side": "worker", "role": role}) for name, role in workers.items()]
    nodes += [(task, task, {"side": "task"}) for task in _TEAM_TASKS]
    edges = [(w, t, float(hours), {"hours": hours}) for w, t, hours in skills]
    code = "teams(contractor=True)" if contractor else "teams()"
    return _build_graph(nodes, edges, False, True, "Teams", code)


# People as name: (club, age group)
_FRIENDS_PEOPLE = {
    "Ada": ("Chess", "20s"),
    "Ben": ("Chess", "20s"),
    "Cleo": ("Chess", "30s"),
    "Dev": ("Chess", "20s"),
    "Ella": ("Chess", "30s"),
    "Femi": ("Chess", "30s"),
    "Gwen": ("Chess", "40s"),
    "Hugo": ("Chess", "30s"),
    "Iris": ("Chess", "40s"),
    "Jules": ("Chess", "40s"),
    "Kemal": ("Chess", "50s"),
    "Lou": ("Chess", "50s"),
    "Mara": ("Climbing", "20s"),
    "Nico": ("Climbing", "20s"),
    "Oona": ("Climbing", "20s"),
    "Pablo": ("Climbing", "30s"),
    "Rosa": ("Climbing", "20s"),
    "Sven": ("Climbing", "30s"),
    "Tara": ("Climbing", "30s"),
    "Uma": ("Climbing", "30s"),
    "Vik": ("Climbing", "40s"),
    "Wes": ("Climbing", "20s"),
    "Yara": ("Climbing", "30s"),
    "Zane": ("Climbing", "40s"),
}

# Friendships as (person, person, year they became friends)
_FRIENDS_TIES = [
    # The chess club, in two circles
    ("Ada", "Ben", 2019),
    ("Ada", "Cleo", 2020),
    ("Ben", "Cleo", 2019),
    ("Ben", "Dev", 2021),
    ("Cleo", "Dev", 2022),
    ("Ada", "Ella", 2021),
    ("Dev", "Ella", 2020),
    ("Ella", "Femi", 2018),
    ("Dev", "Femi", 2023),
    ("Femi", "Gwen", 2022),
    ("Gwen", "Iris", 2018),
    ("Gwen", "Kemal", 2019),
    ("Iris", "Kemal", 2020),
    ("Iris", "Jules", 2021),
    ("Jules", "Kemal", 2022),
    ("Kemal", "Lou", 2018),
    ("Jules", "Lou", 2019),
    ("Gwen", "Lou", 2024),
    # The climbing club, in two circles
    ("Mara", "Nico", 2020),
    ("Mara", "Oona", 2019),
    ("Nico", "Oona", 2021),
    ("Nico", "Pablo", 2022),
    ("Oona", "Pablo", 2020),
    ("Mara", "Rosa", 2022),
    ("Pablo", "Rosa", 2021),
    ("Rosa", "Sven", 2019),
    ("Sven", "Vik", 2018),
    ("Vik", "Yara", 2020),
    ("Sven", "Yara", 2021),
    ("Yara", "Zane", 2019),
    ("Vik", "Zane", 2022),
    ("Uma", "Wes", 2024),
    ("Wes", "Zane", 2020),
    ("Uma", "Zane", 2021),
    # Hugo and Tara, a couple, and the friends they share across both clubs
    ("Hugo", "Tara", 2017),
    ("Hugo", "Ben", 2020),
    ("Tara", "Ben", 2021),
    ("Hugo", "Jules", 2018),
    ("Tara", "Jules", 2019),
    ("Hugo", "Oona", 2022),
    ("Tara", "Oona", 2018),
    ("Hugo", "Wes", 2021),
    ("Tara", "Wes", 2020),
]

# The friendships made in the year after the first snapshot
_FRIENDS_NEW_TIES = [
    ("Ada", "Dev", 2025),
    ("Gwen", "Jules", 2025),
    ("Mara", "Pablo", 2025),
    ("Sven", "Zane", 2025),
    ("Cleo", "Ella", 2025),
    ("Nico", "Rosa", 2025),
    ("Lou", "Iris", 2025),
    ("Femi", "Vik", 2025),
]


def friends(snapshot: int = 1) -> Graph:
    """
    A friendship network in two snapshots, a year apart.

    The friendships among 24 members of two clubs in a made-up town, a chess club and a
    climbing club, at the end of 2024 (`snapshot=1`) and at the end of 2025 (`snapshot=2`),
    when 8 new friendships have formed. It's the example for link prediction (score the pairs
    in the first snapshot, then check the scores against the friendships that formed), for
    similarity and community detection, and for comparing two versions of a network.

    Network Properties:

    - 24 nodes (people, labeled by name)
    - 43 edges (friendships) in the first snapshot, 51 in the second
    - Undirected, unweighted
    - Connected

    Node Attributes:

    - `club`: `"Chess"` or `"Climbing"`
    - `age_group`: `"20s"`, `"30s"`, `"40s"` or `"50s"`

    Edge Attributes:

    - `since`: the year the two became friends, up to 2024 in the first snapshot and 2025
      for the new friendships

    What to Look For:

    - **Four circles in two clubs.** Each club splits into two circles of friends, and
      `cmty_louvain()` finds the four circles.
    - **A couple who bridge the clubs.** Hugo (chess) and Tara (climbing) have been together
      since 2017 and share four friends, two from each club, who don't know each other. That
      gives their tie the highest dispersion in the network, the signature of a romantic tie.
    - **Most new friendships close a triangle.** Seven of the eight new friendships join two
      people with friends in common, and the Jaccard coefficient ranks those seven among its
      top eight pairs in the first snapshot.
    - **One that no one could predict.** Femi and Vik had no friends in common, so every
      neighborhood-based predictor scores their pair 0.

    Parameters
    ----------
    snapshot
        Which snapshot to load: `1` (default) for the end of 2024, or `2` for a year later,
        with the 8 new friendships added. The second snapshot contains every friendship of the
        first.

    Returns
    -------
    Graph
        The friendship network.

    Raises
    ------
    ValueError
        If `snapshot=` is not `1` or `2`.

    Examples
    --------
    Draw the first snapshot, colored by club:

    ```{python}
    import edgecase as ec

    friends = ec.friends()

    friends.plot(color_by="club", label_by="label", seed=8)
    ```

    The friendships that formed in the following year:

    ```{python}
    import polars as pl

    later = ec.friends(snapshot=2)

    later.get_edge_df().filter(pl.col("since") == 2025).select("source", "target")
    ```
    """
    if snapshot not in (1, 2):
        raise ValueError(f"snapshot= must be 1 or 2, not {snapshot!r}")
    ties = _FRIENDS_TIES + (_FRIENDS_NEW_TIES if snapshot == 2 else [])
    nodes = [
        (name, name, {"club": club, "age_group": age})
        for name, (club, age) in _FRIENDS_PEOPLE.items()
    ]
    edges = [(u, v, None, {"since": year}) for u, v, year in ties]
    return _build_graph(nodes, edges, False, False, "Friends", f"friends(snapshot={snapshot})")


# Papers as "Author year": (title, field, y), drawn as a timeline: x is the year and y keeps the
# learning papers above the networks papers
_PAPERS = {
    "Moreau 2000": ("Random walks on sparse graphs", "networks", 0.0),
    "Nakamura 2001": ("Kernel methods for structured data", "learning", 5.8),
    "Tanaka 2002": ("Hitting times and graph structure", "networks", -3.4),
    "Silva 2003": ("Spectral bounds on mixing", "networks", 1.6),
    "Adeyemi 2003": ("Margins and generalization", "learning", 8.4),
    "Petrov 2004": ("Learning with similarity matrices", "learning", 5.0),
    "Okafor 2005": ("Ranking nodes by stationary flow", "networks", -1.4),
    "Lindqvist 2006": ("Communities from random walks", "networks", -4.6),
    "Garcia 2006": ("Semi-supervised learning by propagation", "learning", 7.6),
    "Brandt 2008": ("Fast approximate ranking", "networks", 1.2),
    "Novak 2008": ("Deep feature hierarchies", "learning", 9.6),
    "Kowalski 2009": ("Diffusion on weighted networks", "networks", -6.2),
    "Chen 2010": ("Representation learning at scale", "learning", 6.0),
    "Haddad 2011": ("Local clustering with personalized walks", "networks", -1.8),
    "Dubois 2012": ("Regularizing with neighborhoods", "learning", 8.8),
    "Ferreira 2013": ("Random walks on temporal networks", "networks", -5.0),
    "Iyer 2014": ("Learning node embeddings from random walks", "networks", 3.4),
    "Rossi 2016": ("Embedding signed networks", "networks", -3.4),
    "Kim 2017": ("Link prediction with walk embeddings", "networks", 0.2),
    "Weber 2019": ("Scalable graph embeddings", "networks", 2.6),
    "Hughes 2020": ("Random walks on networks: a survey", "networks", -2.0),
    "Osei 2021": ("Open problems in network sampling", "networks", -0.2),
}

# Each paper's reference list
_REFERENCES = {
    "Tanaka 2002": ["Moreau 2000"],
    "Silva 2003": ["Moreau 2000", "Tanaka 2002"],
    "Adeyemi 2003": ["Nakamura 2001"],
    "Petrov 2004": ["Nakamura 2001", "Adeyemi 2003"],
    "Okafor 2005": ["Moreau 2000", "Silva 2003"],
    "Lindqvist 2006": ["Moreau 2000", "Tanaka 2002", "Okafor 2005"],
    "Garcia 2006": ["Nakamura 2001", "Petrov 2004"],
    "Brandt 2008": ["Moreau 2000", "Okafor 2005"],
    "Novak 2008": ["Adeyemi 2003", "Garcia 2006"],
    "Kowalski 2009": ["Moreau 2000", "Silva 2003", "Lindqvist 2006"],
    "Chen 2010": ["Nakamura 2001", "Petrov 2004", "Novak 2008"],
    "Haddad 2011": ["Okafor 2005", "Lindqvist 2006", "Brandt 2008"],
    "Dubois 2012": ["Garcia 2006", "Chen 2010"],
    "Ferreira 2013": ["Moreau 2000", "Kowalski 2009", "Haddad 2011"],
    "Iyer 2014": ["Moreau 2000", "Lindqvist 2006", "Haddad 2011", "Garcia 2006", "Chen 2010"],
    "Rossi 2016": ["Iyer 2014", "Ferreira 2013"],
    "Kim 2017": ["Iyer 2014", "Haddad 2011", "Rossi 2016"],
    "Weber 2019": ["Iyer 2014", "Brandt 2008", "Kim 2017"],
    "Hughes 2020": [
        "Moreau 2000",
        "Tanaka 2002",
        "Silva 2003",
        "Okafor 2005",
        "Lindqvist 2006",
        "Brandt 2008",
        "Kowalski 2009",
        "Haddad 2011",
        "Ferreira 2013",
        "Iyer 2014",
        "Kim 2017",
    ],
    "Osei 2021": ["Hughes 2020", "Weber 2019", "Ferreira 2013"],
}


def citations() -> Graph:
    """
    Papers in two research fields and the papers they cite.

    Twenty-two made-up papers from two fields, the study of networks and machine learning,
    with an edge from each paper to every paper it cites. A paper can only cite older ones, so
    the graph is a directed acyclic graph (DAG) ordered by year. It's the example for the
    directed measures of importance (PageRank, hubs and authorities, Katz centrality), for
    reachability, and for hierarchy in directed graphs.

    Network Properties:

    - 22 nodes (papers, labeled by first author and year, such as `"Moreau 2000"`)
    - 58 edges (from a citing paper to the paper it cites)
    - Directed, acyclic, unweighted

    Node Attributes:

    - `title`: the paper's title
    - `field`: `"networks"` (15 papers) or `"learning"` (7 papers)
    - `year`: the year it was published, from 2000 to 2021
    - `pos`: an `(x, y)` position for drawing the papers as a timeline, with `x` the year and the
      learning papers above the networks papers

    What to Look For:

    - **A seminal paper.** Moreau 2000 is cited by 9 papers, more than any other, and is the
      top authority and the top paper by PageRank.
    - **A survey.** Hughes 2020 cites 11 papers, most of them the field's best-cited, so it's
      the top hub by a wide margin, although only one paper cites it.
    - **One paper joins the fields.** Iyer 2014 is the only paper to cite work in both fields,
      so it's the one articulation point of the undirected view, and has the highest
      betweenness.
    - **PageRank sees both fields; authority sees one.** Nakamura 2001, the learning field's
      founding paper, is second by PageRank but only eighteenth by authority score: the hubs
      that make a paper an authority are all networks papers.

    Returns
    -------
    Graph
        The citation network.

    Examples
    --------
    Draw the papers as a timeline, colored by field, with every citation pointing back in
    time. Iyer 2014 is the one paper with references in both fields:

    ```{python}
    import edgecase as ec

    citations = ec.citations()

    citations.plot(layout="pos", color_by="field", label_by="label")
    ```

    The best-cited papers are the authorities, and the papers that cite them the hubs:

    ```{python}
    (
        citations.authority_centrality()
        .hub_centrality()
        .get_node_df()
        .sort("authority_centrality", descending=True)
        .select("label", "year", "authority_centrality", "hub_centrality")
        .head(5)
    )
    ```
    """
    nodes = [
        (
            paper,
            paper,
            {"title": title, "field": field, "year": int(paper[-4:]), "pos": (int(paper[-4:]), y)},
        )
        for paper, (title, field, y) in _PAPERS.items()
    ]
    edges = [(paper, cited, None, {}) for paper, refs in _REFERENCES.items() for cited in refs]
    return _build_graph(nodes, edges, True, False, "Citations", "citations()")


# Currencies as code: (name, value in US dollars)
_CURRENCIES = {
    "USD": ("US dollar", 1.0),
    "EUR": ("Euro", 1.08),
    "GBP": ("British pound", 1.27),
    "JPY": ("Japanese yen", 1 / 150),
    "CHF": ("Swiss franc", 1.12),
    "CAD": ("Canadian dollar", 0.73),
}

# Traded pairs as (currency, currency, spread in percent); each pair is quoted both ways, and
# a quote gives the fair rate less the spread
_CURRENCY_PAIRS = [
    ("USD", "EUR", 0.1),
    ("USD", "GBP", 0.1),
    ("USD", "JPY", 0.1),
    ("USD", "CHF", 0.1),
    ("USD", "CAD", 0.1),
    ("EUR", "GBP", 0.15),
    ("EUR", "JPY", 0.15),
    ("EUR", "CHF", 0.15),
    ("JPY", "CHF", 0.5),
]

# The stale euro-pound quotes price the pound this much too cheap in euros
_CURRENCY_STALE = 1.0045


def currency(arbitrage: bool = True) -> Graph:
    """
    Exchange rates between six currencies, with one arbitrage loop.

    The rates quoted one afternoon at a made-up currency desk, for the shortest-path methods
    that allow negative weights. Each edge is a quote, directed from the currency sold to the
    currency bought, and its `rate` is how many units of the second one unit of the first buys.
    Converting along a path multiplies the rates, so the edges are weighted by the negative
    logarithm of the rate: the path of least total weight is the best way to convert, and a
    rate above 1 is a negative weight. A cycle of negative total weight is an arbitrage: a
    round trip that ends with more money than it started with. The euro–pound quotes are out
    of date and make one such loop; set `arbitrage=False` to correct them.

    Network Properties:

    - 6 nodes (currencies, labeled by ISO code, such as `"USD"`)
    - 18 edges (9 traded pairs, each quoted both ways)
    - Directed, weighted by the negative log of the rate, with negative weights
    - Strongly connected

    Node Attributes:

    - `name`: the currency's name, such as `"Swiss franc"`

    Edge Attributes:

    - `rate`: the units of the target currency bought by one unit of the source currency
    - `neg_log_rate`: $-\\ln(\\text{rate})$, the same as the edge weight

    What to Look For:

    - **One arbitrage loop.** Dollars changed to euros, then pounds, then back to dollars,
      return 1.00099 dollars for each dollar: the only cycle of negative total weight. Every
      other round trip loses money to the spreads.
    - **Negative weights.** Every rate above 1 (a pound buys 1.26873 dollars) is a negative
      weight, so Dijkstra's algorithm doesn't apply; Bellman-Ford, Johnson and Goldberg-Radzik
      do, once `arbitrage=False` removes the loop.
    - **The direct route isn't the best.** Few people trade yen for Swiss francs, so the direct
      quote loses 0.5% to the spread, while going through the dollar loses 0.2%.
    - **A hub and a spoke.** The dollar trades against every other currency, and the Canadian
      dollar against the US dollar only, so every route to or from it goes through `"USD"`.

    Parameters
    ----------
    arbitrage
        Whether the euro–pound quotes are the stale ones that make the arbitrage loop. Default is
        `True`. With `False`, they're corrected and no cycle has negative total weight.

    Returns
    -------
    Graph
        The exchange-rate network.

    Examples
    --------
    Some cycle has negative total weight, and the one reachable from the dollar is the
    arbitrage loop:

    ```{python}
    import edgecase as ec

    fx = ec.currency()
    loop = fx.find_negative_cycle("USD")

    fx.negative_edge_cycle(), [fx.get_label_from_id(n) for n in loop]
    ```

    Multiplying the rates around the loop gives the gain on each unit of currency, and
    drawing the loop's quotes highlighted shows where it runs:

    ```{python}
    import math

    edges = fx.get_edge_df()
    pairs = list(zip(edges["source"], edges["target"]))
    rate = dict(zip(pairs, edges["rate"]))
    edge_id = dict(zip(pairs, edges["id"]))

    steps = list(zip(loop, loop[1:]))
    round(math.prod(rate[step] for step in steps), 5)
    ```

    ```{python}
    fx.plot(
        highlight_edges=[edge_id[step] for step in steps],
        label_by="label", curved_edges=True, seed=4,
    )
    ```

    With the stale quotes corrected, the best way to change yen into Swiss francs goes through
    the dollar:

    ```{python}
    fx = ec.currency(arbitrage=False)

    path = fx.bellman_ford_shortest_path("JPY", "CHF", return_list=True)
    [fx.get_label_from_id(n) for n in path]
    ```
    """
    nodes = [(code, code, {"name": name}) for code, (name, _) in _CURRENCIES.items()]
    edges = []
    for first, second, spread in _CURRENCY_PAIRS:
        for source, target in ((first, second), (second, first)):
            rate = _CURRENCIES[source][1] / _CURRENCIES[target][1] * (1 - spread / 100)
            if arbitrage and (source, target) == ("EUR", "GBP"):
                rate *= _CURRENCY_STALE
            if arbitrage and (source, target) == ("GBP", "EUR"):
                rate /= _CURRENCY_STALE
            rate = float(f"{rate:.6g}")
            weight = -math.log(rate)
            edges.append((source, target, weight, {"rate": rate, "neg_log_rate": weight}))
    code = "currency()" if arbitrage else "currency(arbitrage=False)"
    return _build_graph(nodes, edges, True, True, "Currency", code)


# Towns as name: (x, y, population), with x and y in kilometers from Ashby
_TOWNS = {
    "Ashby": (0.0, 0.0, 12000),
    "Bramford": (22.0, 8.0, 8500),
    "Calder": (40.0, 30.0, 54000),
    "Dunholt": (12.0, 38.0, 6200),
    "Elmwick": (30.0, 58.0, 9100),
    "Fenmoor": (58.0, 62.0, 4300),
    "Greyford": (70.0, 40.0, 15800),
    "Halston": (62.0, 12.0, 7400),
    "Kestrel Bay": (88.0, 6.0, 21000),
    "Larkhill": (96.0, 34.0, 3900),
    "Millbrook": (84.0, 66.0, 11200),
    "Norbury": (46.0, 84.0, 2800),
}

# Roads as (town, town, kilometers by road)
_ROADS = [
    ("Ashby", "Bramford", 26),
    ("Ashby", "Dunholt", 50),
    ("Bramford", "Calder", 33),
    ("Bramford", "Dunholt", 40),
    ("Bramford", "Halston", 47),
    ("Calder", "Dunholt", 34),
    ("Calder", "Elmwick", 37),
    ("Calder", "Fenmoor", 42),
    ("Calder", "Greyford", 38),
    ("Calder", "Halston", 31),
    ("Dunholt", "Elmwick", 30),
    ("Elmwick", "Fenmoor", 32),
    ("Elmwick", "Greyford", 53),
    ("Elmwick", "Norbury", 37),
    ("Fenmoor", "Greyford", 28),
    ("Fenmoor", "Millbrook", 32),
    ("Fenmoor", "Norbury", 28),
    ("Greyford", "Halston", 34),
    ("Greyford", "Kestrel Bay", 48),
    ("Greyford", "Larkhill", 30),
    ("Greyford", "Millbrook", 36),
    ("Halston", "Kestrel Bay", 31),
    ("Halston", "Larkhill", 51),
    ("Kestrel Bay", "Larkhill", 32),
    ("Larkhill", "Millbrook", 39),
    ("Millbrook", "Norbury", 46),
]


def towns() -> Graph:
    """
    Towns and the roads between them, with distances.

    Twelve towns in a made-up county and the 26 roads that join them, weighted by their length
    in kilometers. It's the example for routing problems over a road map: the shortest route
    between two towns, the shortest round trip that visits every town (the traveling salesman
    problem), the cheapest set of roads that keeps every town connected, and the cheapest that
    connects just a few of them (a Steiner tree). Every town has map coordinates, so
    `plot(layout="pos")` draws it as a map.

    Network Properties:

    - 12 nodes (towns, labeled by name)
    - 26 edges (roads)
    - Undirected, weighted by length in kilometers
    - Connected

    Node Attributes:

    - `population`: the number of people who live in the town
    - `pos`: the town's `(x, y)` position on the map, in kilometers from Ashby

    Edge Attributes:

    - `km`: the length of the road, the same as the edge weight

    What to Look For:

    - **Greedy goes the long way round.** The shortest round trip through every town is 396
      km, and visits each town once. `traveling_salesman(method="greedy")`, which always heads
      for the nearest town not yet visited, strands itself and has to double back: its tour is
      533 km. Christofides' algorithm finds one of 429 km, and threshold accepting, with
      `seed=1`, finds the best.
    - **A* has a heuristic.** Every road is longer than the straight line between its towns,
      so the straight-line distance, in kilometers, never overestimates the distance left to
      go.
    - **A Steiner tree.** The cheapest roads joining Ashby, Elmwick and Halston go through
      Bramford and Calder, 127 km in all.
    - **The big town isn't the busiest junction.** Calder, with 54,000 people, is the largest
      town, but more roads meet at Greyford: 7, against Calder's 6.
    - **A minimum spanning tree.** The cheapest set of roads that keeps every town connected
      is 333 km long, against 965 km for all 26.

    Returns
    -------
    Graph
        The road network.

    Examples
    --------
    Draw the towns as a map, sized by population:

    ```{python}
    import edgecase as ec

    towns = ec.towns()

    towns.plot(layout="pos", size_by="population", label_by="label")
    ```

    The shortest round trip through every town, found by threshold accepting:

    ```{python}
    tour = towns.traveling_salesman(method="threshold_accepting", seed=1)

    towns.plot(layout="pos", highlight=tour, label_by="label")
    ```
    """
    nodes = [
        (name, name, {"population": population, "pos": (x, y)})
        for name, (x, y, population) in _TOWNS.items()
    ]
    edges = [(u, v, float(km), {"km": km}) for u, v, km in _ROADS]
    return _build_graph(nodes, edges, False, True, "Towns", "towns()")


# Rooms as name: (floor, kind, x, y), with x and y in meters; floor 2 is drawn above floor 1
_ROOMS = {
    "Lobby": (1, "common", 3.0, 0.0),
    "Reception": (1, "common", 0.0, 4.0),
    "West Hall 1": (1, "hall", 6.0, 4.0),
    "East Hall 1": (1, "hall", 18.0, 4.0),
    "Stairs 1": (1, "stairs", 24.0, 4.0),
    "Kitchen": (1, "common", 3.0, 8.0),
    "Meeting A": (1, "meeting", 9.0, 8.0),
    "Meeting B": (1, "meeting", 15.0, 8.0),
    "Office 101": (1, "office", 21.0, 0.0),
    "Office 102": (1, "office", 21.0, 8.0),
    "Library": (2, "common", 3.0, 12.0),
    "Lounge": (2, "common", 12.0, 12.0),
    "Server Room": (2, "server", 21.0, 12.0),
    "West Hall 2": (2, "hall", 6.0, 16.0),
    "East Hall 2": (2, "hall", 18.0, 16.0),
    "Stairs 2": (2, "stairs", 24.0, 16.0),
    "Office 201": (2, "office", 3.0, 20.0),
    "Office 202": (2, "office", 9.0, 20.0),
    "Office 203": (2, "office", 15.0, 20.0),
    "Office 204": (2, "office", 21.0, 20.0),
}

# Doors between rooms, as (room, room)
_DOORS = [
    ("Lobby", "Reception"),
    ("Lobby", "West Hall 1"),
    ("Reception", "West Hall 1"),
    ("Kitchen", "West Hall 1"),
    ("West Hall 1", "East Hall 1"),
    ("West Hall 1", "Meeting A"),
    ("Meeting A", "Meeting B"),
    ("Meeting B", "East Hall 1"),
    ("Office 101", "East Hall 1"),
    ("Office 102", "East Hall 1"),
    ("Stairs 1", "East Hall 1"),
    ("Stairs 2", "East Hall 2"),
    ("West Hall 2", "East Hall 2"),
    ("Office 201", "West Hall 2"),
    ("Office 202", "West Hall 2"),
    ("Library", "West Hall 2"),
    ("Library", "Lounge"),
    ("Lounge", "East Hall 2"),
    ("Office 203", "East Hall 2"),
    ("Office 204", "East Hall 2"),
    ("Server Room", "East Hall 2"),
    ("Office 203", "Office 204"),
]


def offices() -> Graph:
    """
    The rooms of a two-story office and the doors between them.

    The floor plan of a made-up office, as a graph of its 20 rooms, joined where a door leads
    from one to the other, and the staircase between the two floors. It's the example for
    covering problems on a building: the fewest rooms from which every room can be watched
    (a dominating set), the most rooms with no door between any two (an independent set), and
    coloring the rooms so no two next to each other match. Every room has a position on the
    plan, so `plot(layout="pos")` draws the two floors, one above the other.

    Network Properties:

    - 20 nodes (rooms, labeled by name)
    - 23 edges (22 doors and the staircase)
    - Undirected, unweighted
    - Connected and planar

    Node Attributes:

    - `floor`: `1` or `2`
    - `kind`: `"common"` (lobby, reception, kitchen, library and lounge), `"hall"`,
      `"stairs"`, `"meeting"`, `"office"` or `"server"`
    - `pos`: the room's `(x, y)` position on the plan, in meters, with the second floor drawn
      above the first

    Edge Attributes:

    - `kind`: `"door"`, or `"stairs"` for the staircase

    What to Look For:

    - **Four halls watch every room.** Every room is one of the four halls or has a door into
      one, and no three rooms manage that, so the halls are the smallest dominating set, and
      the only one of that size. `minimum_weighted_dominating_set()` finds them, while the
      quicker `dominating_set()` returns a dominating set of 10 rooms.
    - **One way between floors.** The staircase is the only connection between the floors,
      so it's a bridge, and both stairwells are articulation points. So are the four halls,
      since some rooms open onto nothing else.
    - **Three colors.** The Lobby, Reception and West Hall 1 all open onto each other, so
      coloring the rooms takes at least 3 colors, and DSATUR manages with 3.
    - **Eleven rooms with no door between them.** The largest independent set has 11 of the
      20 rooms.

    Returns
    -------
    Graph
        The floor plan.

    Examples
    --------
    Draw the plan, colored by the kind of room:

    ```{python}
    import edgecase as ec

    offices = ec.offices()

    offices.plot(layout="pos", color_by="kind", label_by="label")
    ```

    Every room is one of the four halls, or next to one:

    ```{python}
    halls = offices.minimum_weighted_dominating_set()

    offices.plot(layout="pos", highlight=halls, label_by="label")
    ```
    """
    nodes = [
        (room, room, {"floor": floor, "kind": kind, "pos": (x, y)})
        for room, (floor, kind, x, y) in _ROOMS.items()
    ]
    edges = [(u, v, None, {"kind": "door"}) for u, v in _DOORS]
    edges.append(("Stairs 1", "Stairs 2", None, {"kind": "stairs"}))
    return _build_graph(nodes, edges, False, False, "Offices", "offices()")


# Tasks as name: (days, owner, x, y); Start and Finish are milestones that take no time. The
# positions draw the plan left to right, with the critical path along the middle
_TASKS = {
    "Start": (0, None, 0.0, 0.0),
    "Sign lease": (5, "Rosa", 1.0, 0.0),
    "Design": (10, "Theo", 2.0, 0.0),
    "Permits": (15, "Rosa", 3.0, 0.0),
    "Order equipment": (20, "Imani", 3.5, -1.2),
    "Build out": (25, "Theo", 4.0, 0.0),
    "Install": (4, "Imani", 5.0, 0.0),
    "Plan menu": (8, "Kenji", 2.0, 1.2),
    "Source suppliers": (6, "Kenji", 3.5, 1.2),
    "Hire staff": (12, "Rosa", 2.0, -1.2),
    "Training": (5, "Kenji", 6.0, 0.0),
    "Launch website": (7, "Imani", 3.0, 2.4),
    "Marketing": (14, "Theo", 4.5, 2.4),
    "Health inspection": (3, "Rosa", 5.9, 1.2),
    "Soft opening": (2, "Kenji", 7.0, 0.0),
    "Finish": (0, None, 8.0, 0.0),
}

# Dependencies as (task, task that can't start until it's done)
_DEPENDENCIES = [
    ("Start", "Sign lease"),
    ("Sign lease", "Design"),
    ("Sign lease", "Plan menu"),
    ("Sign lease", "Hire staff"),
    ("Design", "Permits"),
    ("Design", "Order equipment"),
    ("Permits", "Build out"),
    ("Build out", "Install"),
    ("Order equipment", "Install"),
    ("Plan menu", "Source suppliers"),
    ("Plan menu", "Launch website"),
    ("Launch website", "Marketing"),
    ("Install", "Health inspection"),
    ("Source suppliers", "Health inspection"),
    ("Hire staff", "Training"),
    ("Install", "Training"),
    ("Health inspection", "Soft opening"),
    ("Training", "Soft opening"),
    ("Marketing", "Soft opening"),
    ("Soft opening", "Finish"),
]


def project() -> Graph:
    """
    The tasks in opening a café, and the order they must be done in.

    The project plan for opening a made-up café, from signing the lease to the soft opening,
    with an edge from each task to every task that can't start until it's done. The graph is a
    directed acyclic graph (DAG), and each edge is weighted by the days the first task takes,
    so the longest path from `"Start"` to `"Finish"` is the critical path: the chain of tasks
    that sets how soon the café can open, where any delay delays the opening.

    Network Properties:

    - 16 nodes (14 tasks, and the milestones `"Start"` and `"Finish"`, labeled by name)
    - 20 edges (from a task to a task that depends on it)
    - Directed, acyclic, weighted by the days the first task takes

    Node Attributes:

    - `days`: how many days the task takes (`0` for the milestones)
    - `owner`: who is responsible for it, `"Rosa"`, `"Theo"`, `"Imani"` or `"Kenji"` (missing
      for the milestones)
    - `pos`: an `(x, y)` position for drawing the plan left to right, with the critical path
      along the middle

    Edge Attributes:

    - `days`: how many days the first task takes, the same as the edge weight

    What to Look For:

    - **A 66-day critical path.** The longest path runs from signing the lease through the
      design, the permits, the build-out, installing the equipment and training the staff to
      the soft opening, and takes 66 days. `dag_longest_path()` finds it.
    - **Slack elsewhere.** Every task off the critical path can slip without delaying the
      opening: ordering the equipment by up to 20 days, the health inspection by 2, and
      hiring staff by 42.
    - **Nine stages.** The topological generations put the tasks in 9 stages, each of which
      can start once the one before it is done.
    - **Bottlenecks.** Every path from the start to installing the equipment goes through the
      design, its immediate dominator, and every path to the soft opening through signing
      the lease.

    Returns
    -------
    Graph
        The project plan.

    Examples
    --------
    Find the critical path and draw it on the plan:

    ```{python}
    import edgecase as ec

    project = ec.project()
    critical = project.dag_longest_path()

    project.plot(
        layout="pos", highlight=critical, label_by="label", height=360
    )
    ```

    Its length is the fewest days the project can take:

    ```{python}
    project.dag_longest_path_length()
    ```
    """
    nodes = [
        (task, task, {"days": days, "owner": owner, "pos": (x, y)})
        for task, (days, owner, x, y) in _TASKS.items()
    ]
    edges = [(u, v, float(_TASKS[u][0]), {"days": _TASKS[u][0]}) for u, v in _DEPENDENCIES]
    return _build_graph(nodes, edges, True, True, "Project", "project()")


# Variables as name: (observed, x, y)
_CAUSAL_VARIABLES = {
    "Age": (True, 0.0, 1.5),
    "Genetics": (False, 3.75, -1.5),
    "Exercise": (True, 1.0, 0.5),
    "Diet": (True, 1.0, -1.5),
    "Weight": (True, 2.0, -0.5),
    "Medication": (True, 2.0, 2.0),
    "Blood pressure": (True, 3.0, 0.5),
    "Heart disease": (True, 4.5, 0.5),
}

# Causes as (cause, effect)
_CAUSES = [
    ("Age", "Exercise"),
    ("Age", "Medication"),
    ("Age", "Blood pressure"),
    ("Exercise", "Weight"),
    ("Diet", "Weight"),
    ("Weight", "Blood pressure"),
    ("Medication", "Blood pressure"),
    ("Blood pressure", "Heart disease"),
    ("Genetics", "Blood pressure"),
    ("Genetics", "Heart disease"),
]


def causal() -> Graph:
    """
    A causal diagram of the risks for heart disease.

    A simplified causal model of heart disease, as a directed acyclic graph (DAG) with an edge
    from each cause to its effect: age, exercise and diet act on weight and blood pressure,
    and high blood pressure leads to heart disease. One variable, a genetic risk, can't be
    measured. It's the example for reading a causal diagram: which variables are
    independent of which, what to adjust for to estimate an effect, and when no adjustment
    will do. The model is a textbook illustration, not medical advice.

    Network Properties:

    - 8 nodes (variables, labeled by name)
    - 10 edges (from cause to effect)
    - Directed, acyclic, unweighted

    Node Attributes:

    - `observed`: whether the variable can be measured (`False` only for `"Genetics"`)
    - `pos`: an `(x, y)` position for drawing the diagram, with causes to the left of their
      effects

    What to Look For:

    - **A collider.** Exercise and Diet both affect Weight, and nothing else connects them,
      so they're d-separated, and independent, given nothing; but given Weight they're not,
      since among people of the same weight, those who exercise less must eat better.
    - **A confounder.** Age affects both Exercise and Blood pressure, so to estimate the
      effect of exercise on blood pressure, adjust for Age: it's the smallest set that blocks
      every backdoor path.
    - **A mediator.** Exercise affects Blood pressure only through Weight.
    - **An unobserved confounder.** Genetics affects both Blood pressure and Heart disease,
      but can't be measured, so no set of observed variables blocks the backdoor path between
      them, and `find_minimal_d_separator()` returns `None` when restricted to the observed
      ones.
    - **Colliders and v-structures.** `colliders()` finds 8 colliders, and `v_structures()`
      the 6 of them whose parents aren't joined: Age and Medication both affect Blood
      pressure, but Age also affects Medication.

    Returns
    -------
    Graph
        The causal diagram.

    Examples
    --------
    Draw the diagram, with the unobserved variable in its own color:

    ```{python}
    import edgecase as ec

    causal = ec.causal()

    causal.plot(
        layout="pos", color_by="observed", label_by="label", height=360
    )
    ```

    Exercise and Diet are independent, until Weight is known:

    ```{python}
    (
        causal.is_d_separator(["Exercise"], ["Diet"], []),
        causal.is_d_separator(["Exercise"], ["Diet"], ["Weight"]),
    )
    ```
    """
    nodes = [
        (name, name, {"observed": observed, "pos": (x, y)})
        for name, (observed, x, y) in _CAUSAL_VARIABLES.items()
    ]
    edges = [(u, v, None, {}) for u, v in _CAUSES]
    return _build_graph(nodes, edges, True, False, "Causal", "causal()")


# Molecules as name: (atoms, bonds), with hydrogen left out; bonds are (atom, atom, bond type),
# with the atoms numbered from 0 in the order listed
_MOLECULES = {
    "benzene": ("CCCCCC", [(i, (i + 1) % 6, "aromatic") for i in range(6)]),
    "cyclohexane": ("CCCCCC", [(i, (i + 1) % 6, "single") for i in range(6)]),
    "ethanol": ("CCO", [(0, 1, "single"), (1, 2, "single")]),
    "dimethyl_ether": ("COC", [(0, 1, "single"), (1, 2, "single")]),
    "butane": ("CCCC", [(0, 1, "single"), (1, 2, "single"), (2, 3, "single")]),
    "isobutane": ("CCCC", [(0, 1, "single"), (1, 2, "single"), (1, 3, "single")]),
    # From the SMILES string CN1C=NC2=C1C(=O)N(C(=O)N2C)C
    "caffeine": (
        "CNCNCCCONCONCC",
        [
            (0, 1, "single"),
            (1, 2, "single"),
            (2, 3, "double"),
            (3, 4, "single"),
            (4, 5, "double"),
            (5, 1, "single"),
            (5, 6, "single"),
            (6, 7, "double"),
            (6, 8, "single"),
            (8, 9, "single"),
            (9, 10, "double"),
            (9, 11, "single"),
            (11, 4, "single"),
            (11, 12, "single"),
            (8, 13, "single"),
        ],
    ),
}


def molecules() -> dict[str, Graph]:
    """
    Seven small molecules, as graphs of their atoms and bonds.

    Molecular graphs of seven well-known compounds, with a node for each atom and an edge for
    each bond. The hydrogen atoms are left out, as is usual for molecular graphs. They're
    real molecules, chosen in pairs that look alike: ethanol and dimethyl ether have the same
    atoms, as do butane and isobutane, and benzene and cyclohexane have the same shape. That
    makes them the example for graph isomorphism, graph invariants such as the Wiener index,
    and rings.

    Network Properties:

    - `"benzene"`: 6 atoms and 6 bonds, in an aromatic ring
    - `"cyclohexane"`: 6 atoms and 6 single bonds, in a ring
    - `"ethanol"`: 3 atoms (C–C–O) and 2 bonds
    - `"dimethyl_ether"`: 3 atoms (C–O–C) and 2 bonds
    - `"butane"`: 4 carbon atoms in a chain
    - `"isobutane"`: 4 carbon atoms, three of them joined to the fourth
    - `"caffeine"`: 14 atoms (8 carbon, 4 nitrogen, 2 oxygen) and 15 bonds, in two fused rings
    - All undirected and unweighted, and each labeled by element and number, such as `"C1"`
      or `"O2"`

    Node Attributes:

    - `element`: the chemical element, `"C"`, `"N"` or `"O"`

    Edge Attributes:

    - `bond`: `"single"`, `"double"` or `"aromatic"`

    What to Look For:

    - **Same shape, different molecule.** Ethanol and dimethyl ether are both three atoms in a
      row, so `is_isomorphic()` says they match, but not with `node_attr="element"`: the
      oxygen is at the end of one and the middle of the other. Benzene and cyclohexane are
      both rings of six carbons, and only `edge_attr="bond"` tells them apart.
    - **Same atoms, different shape.** Butane and isobutane both have four carbons, but one is
      a chain and the other branched, so they're not isomorphic.
    - **The Wiener index.** The sum of the distances between all pairs of atoms is 10 for
      butane and 9 for the more compact isobutane. Wiener introduced the index in 1947 to
      predict boiling points, which are lower for more compact molecules.
    - **Two rings, or one.** Caffeine's rings have 5 and 6 atoms, and
      `minimum_cycle_basis()` finds those two, but `cycle_basis()` can return the 9-atom
      ring around the outside of both in place of one of them. The girth is 5.

    Returns
    -------
    dict[str, Graph]
        The molecules, keyed by name: `"benzene"`, `"cyclohexane"`, `"ethanol"`,
        `"dimethyl_ether"`, `"butane"`, `"isobutane"` and `"caffeine"`.

    Examples
    --------
    Draw caffeine, colored by element, with its bonds labeled:

    ```{python}
    import edgecase as ec

    caffeine = ec.molecules()["caffeine"]

    caffeine.plot(
        color_by="element", label_by="label", edge_label_by="bond", seed=1
    )
    ```

    Ethanol and dimethyl ether have the same shape, but not with the atoms in the same
    places:

    ```{python}
    molecules = ec.molecules()
    ethanol, ether = molecules["ethanol"], molecules["dimethyl_ether"]

    (
        ethanol.is_isomorphic(ether),
        ethanol.is_isomorphic(ether, node_attr="element"),
    )
    ```
    """
    out = {}
    for name, (atoms, bonds) in _MOLECULES.items():
        counts: dict[str, int] = {}
        labels = []
        for element in atoms:
            counts[element] = counts.get(element, 0) + 1
            labels.append(f"{element}{counts[element]}")
        nodes = [(label, label, {"element": element}) for label, element in zip(labels, atoms)]
        edges = [(labels[a], labels[b], None, {"bond": bond}) for a, b, bond in bonds]
        title = name.replace("_", " ").capitalize()
        out[name] = _build_graph(nodes, edges, False, False, title, f"molecules()[{name!r}]")
    return out


# Land masses as name: (kind, x, y)
_LAND = {
    "Kneiphof": ("island", 0.0, 0.0),
    "North Bank": ("bank", 0.0, 2.0),
    "South Bank": ("bank", 0.0, -2.0),
    "Lomse": ("island", 2.5, 0.0),
}

# Bridges as (name, land, land)
_BRIDGES = [
    ("Krämerbrücke", "Kneiphof", "North Bank"),
    ("Schmiedebrücke", "Kneiphof", "North Bank"),
    ("Holzbrücke", "North Bank", "Lomse"),
    ("Grüne Brücke", "Kneiphof", "South Bank"),
    ("Köttelbrücke", "Kneiphof", "South Bank"),
    ("Hohe Brücke", "South Bank", "Lomse"),
    ("Honigbrücke", "Kneiphof", "Lomse"),
]


def konigsberg() -> Graph:
    """
    The seven bridges of Königsberg.

    The city of Königsberg, in Prussia, was built where the Pregel river runs around two
    islands, and in the 18th century seven bridges joined the islands to each other and to the
    banks. Its people wondered whether they could walk through the city crossing each bridge
    exactly once. In 1736 Leonhard Euler showed they couldn't, and in doing so founded graph
    theory: each land mass is a node and each bridge an edge, and the walk exists only if no
    more than two nodes have an odd number of edges. Unlike the other example networks, this
    one is real.

    Network Properties:

    - 4 nodes (land masses, labeled by name)
    - 7 edges (bridges), with two pairs of parallel edges
    - Undirected, unweighted
    - Connected

    Node Attributes:

    - `kind`: `"island"` (Kneiphof and Lomse) or `"bank"`
    - `pos`: an `(x, y)` position for drawing the city as a map, with north at the top

    Edge Attributes:

    - `bridge`: the bridge's name, such as `"Krämerbrücke"`

    What to Look For:

    - **Every node is odd.** Kneiphof has 5 bridges and the other three land masses 3 each,
      so no walk crosses every bridge exactly once: `is_eulerian()` and
      `has_eulerian_path()` are both `False`.
    - **Two more bridges would do it.** `eulerize()` duplicates two bridges, one between
      Kneiphof and the North Bank and the Hohe Brücke between the South Bank and Lomse, which
      leaves every land mass with an even number and makes the round trip possible.
    - **A multigraph.** Kneiphof is joined to each bank by two bridges, so the graph has
      parallel edges and isn't simple.

    Returns
    -------
    Graph
        The bridges of Königsberg.

    Examples
    --------
    Draw the city as a map, with each bridge labeled:

    ```{python}
    import edgecase as ec

    konigsberg = ec.konigsberg()

    konigsberg.plot(
        layout="pos", color_by="kind", label_by="label",
        edge_label_by="bridge",
    )
    ```

    Every land mass has an odd number of bridges, so no walk crosses each one exactly once:

    ```{python}
    label = konigsberg.get_label_from_id
    degrees = {
        label(n): konigsberg.get_degree(n) for n in konigsberg.get_node_ids()
    }

    degrees, konigsberg.has_eulerian_path()
    ```
    """
    nodes = [(name, name, {"kind": kind, "pos": (x, y)}) for name, (kind, x, y) in _LAND.items()]
    edges = [(u, v, None, {"bridge": bridge}) for bridge, u, v in _BRIDGES]
    return _build_graph(nodes, edges, False, False, "Königsberg", "konigsberg()")


# People as name: (title, department, manager)
_STAFF = {
    "Mira": ("Chief Executive Officer", "Executive", None),
    "Jonas": ("Chief Technology Officer", "Engineering", "Mira"),
    "Leila": ("Chief Financial Officer", "Finance", "Mira"),
    "Sam": ("Sales Director", "Sales", "Mira"),
    "Uma": ("HR Manager", "People", "Mira"),
    "Ines": ("VP of Engineering", "Engineering", "Jonas"),
    "Kwame": ("Engineering Manager", "Engineering", "Ines"),
    "Ruth": ("Tech Lead", "Engineering", "Kwame"),
    "Owen": ("Engineer", "Engineering", "Kwame"),
    "Tariq": ("Engineer", "Engineering", "Ruth"),
    "Pia": ("Engineer", "Engineering", "Ruth"),
    "Noor": ("Intern", "Engineering", "Tariq"),
    "Hiro": ("Controller", "Finance", "Leila"),
    "Ada": ("Accountant", "Finance", "Hiro"),
    "Bea": ("Account Executive", "Sales", "Sam"),
    "Cal": ("Account Executive", "Sales", "Sam"),
    "Dara": ("Account Executive", "Sales", "Sam"),
    "Eli": ("Account Executive", "Sales", "Sam"),
    "Fay": ("Sales Associate", "Sales", "Sam"),
    "Gus": ("Sales Associate", "Sales", "Sam"),
}


def org_chart() -> Graph:
    """
    A company's reporting lines.

    The organization chart of a made-up company of 20 people, with an edge from each manager
    to each person who reports to them. Everyone but the chief executive has exactly one
    manager, so the graph is a rooted tree (an arborescence). It's the example for trees and
    hierarchies: depth, ancestors and descendants, the lowest common ancestor of two people
    (the most junior manager they share), and drawing a tree top-down.

    Network Properties:

    - 20 nodes (people, labeled by first name)
    - 19 edges (from manager to report)
    - Directed, acyclic, unweighted
    - An arborescence rooted at the chief executive, Mira

    Node Attributes:

    - `title`: the person's job title, such as `"Tech Lead"`
    - `dept`: the department, `"Executive"`, `"Engineering"`, `"Finance"`, `"Sales"` or
      `"People"`
    - `level`: how many levels below the chief executive the person is, from `0` to `6`
    - `pos`: an `(x, y)` position for drawing the chart top-down, with each manager centered
      over their reports

    What to Look For:

    - **A deep chain.** Engineering has six levels below the chief executive, down to the
      intern Noor, while Sales has one: `dag_longest_path()` runs from Mira to Noor.
    - **A wide team.** Sam, the sales director, has 6 direct reports, more than anyone else.
    - **Lowest common ancestors.** The most junior manager Noor and Pia share is Ruth, and
      for Noor and Ada, in another department, it's Mira.
    - **The top isn't the middle.** In the undirected view, the people the fewest steps from
      everyone are Jonas and Ines, partway down the long engineering chain, not Mira. Mira
      does have the highest betweenness, since every path between departments goes through
      the chief executive.

    Returns
    -------
    Graph
        The organization chart.

    Examples
    --------
    Draw the chart top-down, colored by department:

    ```{python}
    import edgecase as ec

    org = ec.org_chart()

    org.plot(layout="pos", color_by="dept", label_by="label", height=420)
    ```

    The most junior manager two people share is their lowest common ancestor:

    ```{python}
    org.get_label_from_id(org.lowest_common_ancestor("Noor", "Pia"))
    ```
    """
    reports: dict[str, list[str]] = {name: [] for name in _STAFF}
    for name, (_, _, manager) in _STAFF.items():
        if manager:
            reports[manager].append(name)
    # Leaves go left to right in order, and each manager is centered over their reports
    pos: dict[str, tuple[float, float]] = {}
    level: dict[str, int] = {}
    leaves = 0

    def place(name: str, depth: int) -> float:
        nonlocal leaves
        if reports[name]:
            xs = [place(report, depth + 1) for report in reports[name]]
            x = (xs[0] + xs[-1]) / 2
        else:
            x = float(leaves)
            leaves += 1
        pos[name] = (x, -float(depth))
        level[name] = depth
        return x

    place("Mira", 0)
    nodes = [
        (name, name, {"title": title, "dept": dept, "level": level[name], "pos": pos[name]})
        for name, (title, dept, _) in _STAFF.items()
    ]
    edges = [(manager, name, None, {}) for name, (_, _, manager) in _STAFF.items() if manager]
    return _build_graph(nodes, edges, True, False, "Org Chart", "org_chart()")


# Proteins as name: (pathway, expression, disease gene, drug target, x, y). Expression is a log2
# fold change, diseased against healthy tissue; the uncharacterized ORFs have no pathway
_PROTEINS = {
    "REP1": ("DNA repair", 2.4, True, None, -1.3, 1.1),
    "REP2": ("DNA repair", 2.1, True, None, -2.5, 1.0),
    "REP3": ("DNA repair", 0.3, False, None, -1.5, 2.2),
    "REP4": ("DNA repair", 1.9, False, None, -2.7, 2.2),
    "REP5": ("DNA repair", 2.6, True, None, -1.9, 3.3),
    "REP6": ("DNA repair", 1.7, False, "Drug A", -3.1, 3.5),
    "ORF1": (None, 1.4, False, None, -3.8, 1.5),
    "CYC1": ("Cell cycle", 1.2, False, None, 0.4, 4.0),
    "CYC2": ("Cell cycle", 1.8, True, None, -0.2, 2.8),
    "CYC3": ("Cell cycle", 0.4, False, "Drug A", 0.9, 1.9),
    "CYC4": ("Cell cycle", -0.2, False, None, 2.0, 2.5),
    "CYC5": ("Cell cycle", 0.1, False, None, 1.8, 3.7),
    "SIG1": ("Signaling", 0.2, False, None, 2.8, 1.5),
    "SIG2": ("Signaling", -0.3, False, None, 1.7, 0.6),
    "SIG3": ("Signaling", 0.1, False, None, 3.1, 0.4),
    "SIG4": ("Signaling", -0.1, False, "Drug B", 3.1, -0.7),
    "SIG5": ("Signaling", 0.3, False, None, 1.7, -0.7),
    "SIG6": ("Signaling", -0.4, False, None, 2.5, -1.8),
    "ORF2": (None, 0.0, False, None, 4.2, -0.1),
    "TRN1": ("Transport", 0.0, False, None, 1.6, -2.6),
    "TRN2": ("Transport", -0.2, False, None, 0.6, -1.6),
    "TRN3": ("Transport", 0.6, True, None, 0.7, -3.3),
    "TRN4": ("Transport", 0.2, False, None, -0.5, -3.3),
    "TRN5": ("Transport", -0.1, False, None, -0.4, -1.8),
    "ORF3": (None, -0.1, False, None, 0.1, -4.3),
    "ENZ1": ("Metabolism", 0.1, False, None, -1.4, -1.5),
    "ENZ2": ("Metabolism", -0.3, False, "Drug B", -2.4, -2.2),
    "ENZ3": ("Metabolism", 2.5, False, None, -3.5, -1.3),
    "ENZ4": ("Metabolism", 0.0, False, None, -2.5, -0.9),
    "ENZ5": ("Metabolism", -0.2, False, None, -1.9, 0.0),
    "HSP1": ("Protein folding", 0.5, False, None, 0.0, 0.0),
}

# Interactions as (protein, protein)
_INTERACTIONS = [
    # DNA repair, with ORF1
    ("REP1", "REP2"),
    ("REP1", "REP3"),
    ("REP2", "REP3"),
    ("REP2", "REP4"),
    ("REP3", "REP4"),
    ("REP3", "REP5"),
    ("REP4", "REP5"),
    ("REP5", "REP6"),
    ("REP4", "REP6"),
    ("ORF1", "REP2"),
    ("ORF1", "REP4"),
    # The cell cycle, and its links to DNA repair
    ("CYC1", "CYC2"),
    ("CYC2", "CYC3"),
    ("CYC1", "CYC3"),
    ("CYC3", "CYC4"),
    ("CYC4", "CYC5"),
    ("CYC1", "CYC5"),
    ("REP6", "CYC1"),
    ("REP5", "CYC2"),
    # Signaling, with ORF2
    ("SIG1", "SIG2"),
    ("SIG2", "SIG3"),
    ("SIG3", "SIG4"),
    ("SIG1", "SIG3"),
    ("SIG4", "SIG5"),
    ("SIG5", "SIG6"),
    ("SIG4", "SIG6"),
    ("SIG2", "SIG5"),
    ("SIG1", "CYC4"),
    ("ORF2", "SIG3"),
    ("ORF2", "SIG4"),
    # Transport, with ORF3
    ("TRN1", "TRN2"),
    ("TRN2", "TRN3"),
    ("TRN1", "TRN3"),
    ("TRN3", "TRN4"),
    ("TRN4", "TRN5"),
    ("TRN2", "TRN5"),
    ("TRN1", "SIG6"),
    ("ORF3", "TRN3"),
    ("ORF3", "TRN4"),
    # Metabolism
    ("ENZ1", "ENZ2"),
    ("ENZ2", "ENZ4"),
    ("ENZ1", "ENZ4"),
    ("ENZ4", "ENZ5"),
    ("ENZ3", "ENZ5"),
    ("ENZ3", "ENZ4"),
    ("ENZ3", "ENZ2"),
    ("ENZ1", "TRN4"),
    # The chaperone HSP1 binds proteins in every pathway
    ("HSP1", "REP1"),
    ("HSP1", "REP2"),
    ("HSP1", "CYC3"),
    ("HSP1", "SIG2"),
    ("HSP1", "SIG5"),
    ("HSP1", "TRN2"),
    ("HSP1", "TRN5"),
    ("HSP1", "ENZ1"),
    ("HSP1", "ENZ5"),
]


def proteins() -> Graph:
    """
    A small protein interaction network, with a disease module.

    An invented interactome of 31 proteins, with an edge for each pair known to bind. Most
    belong to one of five pathways and are named for it: REP for DNA repair, CYC for the cell
    cycle, SIG for signaling, TRN for transport and ENZ for metabolism. HSP1 is a chaperone that
    binds proteins in every pathway, and ORF1 to ORF3 are uncharacterized. Five proteins are
    known disease genes, and each protein has a measured change in expression in diseased
    tissue. It's the example for network medicine: growing a disease module from the known
    genes, testing the module for pathway enrichment, smoothing noisy measurements over the
    network, predicting the pathway of an uncharacterized protein, and measuring how close a
    drug's targets sit to a disease. The proteins, the disease and the drugs are made up.

    Network Properties:

    - 31 nodes (proteins, labeled by name)
    - 56 edges (interactions)
    - Undirected, unweighted
    - Connected

    Node Attributes:

    - `pathway`: `"DNA repair"`, `"Cell cycle"`, `"Signaling"`, `"Transport"`,
      `"Metabolism"` or (for HSP1) `"Protein folding"`; missing for ORF1, ORF2 and ORF3
    - `disease_gene`: `True` for the five known disease genes, REP1, REP2, REP5, CYC2 and
      TRN3, and `False` for the rest
    - `expression`: the change in expression in diseased tissue, against healthy tissue, as a
      log2 fold change (`1.0` is twice as much, `-1.0` half as much)
    - `drug_target`: `"Drug A"` or `"Drug B"` for the two proteins each drug binds, and missing
      for the rest
    - `pos`: an `(x, y)` position for drawing the network, with each pathway in a cluster of
      its own around HSP1

    What to Look For:

    - **A disease module.** Four of the five disease genes sit together, among the DNA repair
      proteins. With `min_score=0.5`, `expand_seed_set()` grows those four into a module of 8:
      all six DNA repair proteins, CYC2 and ORF1. Against 6 DNA repair proteins in the whole
      network, `subgraph_enrichment()` puts the chance of that at 4 in 100,000.
    - **A lone disease gene.** TRN3 is three steps from the other four. With a prize of 3
      for each disease gene and a cost of 1 for each edge, `prize_collecting_steiner_tree()`
      connects the four through REP3 and leaves TRN3 out; it joins when edges cost 0.5.
    - **A sticky hub.** HSP1 has 9 partners, the most in the network, two of them disease
      genes. Random walks from the disease genes pass through it often, so
      `random_walk_with_restart()` ranks it above every protein but REP4 that isn't a disease
      gene itself. `expand_seed_set()` scores a protein by the share of its partners inside the
      module, and leaves HSP1 out.
    - **Noisy measurements.** REP3 barely changed (`0.3`), though its four partners all went
      up, and ENZ3 changed more than any protein but REP5 (`2.5`), though none of its partners
      did. `network_smooth()` raises REP3 to `1.08` and nearly halves ENZ3, to `1.35`, below six
      proteins of the module.
    - **Uncharacterized proteins.** From the pathways of the other proteins,
      `label_propagation()` puts ORF1 in DNA repair, ORF2 in signaling and ORF3 in transport:
      the pathways of their partners.
    - **A drug near the disease, and one far from it.** Drug A binds REP6 and CYC3, each next
      to a disease gene. Drug B binds SIG4 and ENZ2, three steps from the nearest one, and
      `network_proximity()` gives it a positive separation from the disease genes (`0.25`)
      where Drug A's is negative (`-0.05`).

    Returns
    -------
    Graph
        The protein interaction network.

    Examples
    --------
    Draw the network, colored by pathway:

    ```{python}
    import edgecase as ec

    proteins = ec.proteins()

    proteins.plot(layout="pos", color_by="pathway", label_by="label")
    ```

    The disease genes, with their changes in expression:

    ```{python}
    import polars as pl

    (
        proteins.get_node_df()
        .filter(pl.col("disease_gene"))
        .select("label", "pathway", "expression")
    )
    ```
    """
    nodes = []
    for name, (pathway, expression, disease_gene, drug, x, y) in _PROTEINS.items():
        attrs = {"expression": expression, "disease_gene": disease_gene, "pos": (x, y)}
        if pathway is not None:
            attrs["pathway"] = pathway
        if drug is not None:
            attrs["drug_target"] = drug
        nodes.append((name, name, attrs))
    edges = [(u, v, None, {}) for u, v in _INTERACTIONS]
    return _build_graph(nodes, edges, False, False, "Proteins", "proteins()")


# Species as name: (kind, x, y), drawn with y at twice the trophic level
_SPECIES = {
    "Clover": ("plant", 0.0, 2.0),
    "Grass": ("plant", 1.8, 2.0),
    "Bramble": ("plant", 4.0, 2.0),
    "Oak": ("plant", 6.0, 2.0),
    "Grasshopper": ("herbivore", 0.8, 4.0),
    "Rabbit": ("herbivore", 2.8, 4.0),
    "Vole": ("herbivore", 4.6, 4.0),
    "Caterpillar": ("herbivore", 6.0, 4.0),
    "Squirrel": ("herbivore", 8.0, 4.0),
    "Jay": ("omnivore", 7.0, 5.0),
    "Frog": ("predator", -0.6, 6.0),
    "Shrew": ("predator", 0.8, 6.0),
    "Weasel": ("predator", 3.8, 6.0),
    "Marten": ("predator", 8.0, 6.0),
    "Hawk": ("predator", 5.0, 7.0),
    "Fox": ("predator", 0.2, 7.333),
}

# Who eats whom, as (food, eater)
_FEEDING = [
    ("Clover", "Grasshopper"),
    ("Grass", "Grasshopper"),
    ("Grass", "Rabbit"),
    ("Bramble", "Rabbit"),
    ("Bramble", "Vole"),
    ("Oak", "Vole"),
    ("Oak", "Caterpillar"),
    ("Oak", "Squirrel"),
    ("Oak", "Jay"),
    ("Caterpillar", "Jay"),
    ("Grasshopper", "Frog"),
    ("Grasshopper", "Shrew"),
    ("Rabbit", "Weasel"),
    ("Vole", "Weasel"),
    ("Squirrel", "Marten"),
    ("Frog", "Fox"),
    ("Shrew", "Fox"),
    ("Rabbit", "Fox"),
    ("Vole", "Hawk"),
    ("Weasel", "Hawk"),
    ("Jay", "Hawk"),
]


def food_web() -> Graph:
    """
    A woodland food web: who eats whom.

    Sixteen species of a made-up woodland, with an edge from each food to each species that
    eats it, so energy flows along the edges from the plants at the bottom to the predators at
    the top. It's the example for the ecology of networks: trophic levels and how coherent they
    are, the species that reach the most of the web, and what a web loses when one species goes.

    Network Properties:

    - 16 nodes (species, labeled by name)
    - 21 edges (from a food to the species that eats it)
    - Directed, acyclic, unweighted

    Node Attributes:

    - `kind`: `"plant"`, `"herbivore"`, `"omnivore"` (the jay, which eats acorns and
      caterpillars) or `"predator"`
    - `pos`: an `(x, y)` position for drawing the web, with the plants at the bottom and each
      animal at a height set by its trophic level

    What to Look For:

    - **Fractional trophic levels.** The plants are at level 1 and the herbivores at level 2.
      Each other species sits one level above the average of what it eats, so the Jay, eating
      acorns and caterpillars, is at 2.5, and the Fox, eating frogs, shrews and rabbits, is at
      3.67, above the Hawk at 3.5.
    - **Top predators.** Nothing eats the Fox, the Hawk or the Marten, so each is an attracting
      component of its own: once energy reaches one, it goes no further.
    - **A keystone plant.** The Caterpillar and the Squirrel eat nothing but the Oak, the
      Marten nothing but squirrels, and the Jay only acorns and caterpillars, so losing the Oak
      starves four more species. Every animal that eats one of the other plants has something
      else to eat.
    - **Plants that feed the most.** The Grass and the Oak each feed 7 of the other 15
      species, directly or through what eats them, so their local reaching centrality is
      `0.467`, the highest in the web.
    - **A tidy hierarchy.** Most feeding links climb one level, but some climb more (the Fox
      eats rabbits, 1.67 levels down), and the trophic incoherence is `0.28`; a web where every
      link climbs exactly one level scores `0`.

    Returns
    -------
    Graph
        The food web.

    Examples
    --------
    Draw the web from the plants up, colored by kind:

    ```{python}
    import edgecase as ec

    web = ec.food_web()

    web.plot(layout="pos", color_by="kind", label_by="label")
    ```

    Each species' trophic level, highest first:

    ```{python}
    (
        web.trophic_levels()
        .get_node_df()
        .select("label", "kind", "trophic_levels")
        .sort("trophic_levels", descending=True)
    )
    ```
    """
    nodes = [(name, name, {"kind": kind, "pos": (x, y)}) for name, (kind, x, y) in _SPECIES.items()]
    edges = [(u, v, None, {}) for u, v in _FEEDING]
    return _build_graph(nodes, edges, True, False, "Food web", "food_web()")


# Basic blocks as name: (code, x, y), drawn left to right from the entry to the exit
_BLOCKS = {
    "entry": ("total, n = 0, 0", -0.4, 0.0),
    "loop": ("for x in values:", 1.0, 0.0),
    "sign test": ("if x < 0:", 2.0, 2.0),
    "negate": ("x = -x", 3.0, 4.0),
    "add": ("total += x\nn += 1", 4.0, 2.0),
    "limit test": ("if total > limit:", 5.0, 0.6),
    "break": ("break", 6.0, 2.0),
    "empty test": ("if n == 0:", 7.0, -1.6),
    "return None": ("return None", 8.0, -3.2),
    "return mean": ("return total / n", 8.0, 0.0),
    "exit": ("", 9.0, -1.6),
}

# Jumps between blocks as (from, to, branch); the branch is missing where control just falls
# through to the next block
_JUMPS = [
    ("entry", "loop", None),
    ("loop", "sign test", "next"),
    ("loop", "empty test", "done"),
    ("sign test", "negate", "true"),
    ("sign test", "add", "false"),
    ("negate", "add", None),
    ("add", "limit test", None),
    ("limit test", "break", "true"),
    ("limit test", "loop", "false"),
    ("break", "empty test", None),
    ("empty test", "return None", "true"),
    ("empty test", "return mean", "false"),
    ("return None", "exit", None),
    ("return mean", "exit", None),
]


def control_flow() -> Graph:
    """
    The control flow graph of a short function.

    The control flow graph a compiler would build for this Python function, which averages the
    absolute values of a list, stopping early once the running total passes a limit:

    ```python
    def mean_abs(values, limit):
        total, n = 0, 0
        for x in values:
            if x < 0:
                x = -x
            total += x
            n += 1
            if total > limit:
                break
        if n == 0:
            return None
        return total / n
    ```

    Each node is a basic block, a run of code that always executes from start to finish, and
    each edge a possible jump from one block to the next. Every run of the function follows a
    path from `entry` to `exit`. It's the example for the analyses compilers do: dominators,
    dominance frontiers, loops and depth-first edge types.

    Network Properties:

    - 11 nodes (basic blocks, labeled by a name for what they do)
    - 14 edges (possible jumps)
    - Directed, unweighted, with one cycle (the loop)

    Node Attributes:

    - `code`: the block's source code, such as `"if x < 0:"`; empty for `exit`
    - `pos`: an `(x, y)` position for drawing the graph, left to right from `entry` to `exit`

    Edge Attributes:

    - `branch`: which way a test went to take the jump, `"true"` or `"false"` for an `if`,
      and `"next"` or `"done"` for the `for` loop; missing where control falls through

    What to Look For:

    - **One loop.** The five blocks from `loop` to `limit test` form the only strongly
      connected component of more than one block, and the jump from `limit test` back to
      `loop` is what closes it.
    - **Dominators.** Every path from `entry` to `add` passes through `sign test`, but not
      through `negate`, which only runs for negative values, so `sign test` is the immediate
      dominator of `add`. And every path to `exit` passes through `empty test`.
    - **Where paths merge.** The dominance frontier of `negate` is `add`, where the paths with
      and without `x = -x` meet again: the place a compiler in SSA form puts a φ-function for
      `x`. The frontier of `loop` is `loop` itself, the sign of a loop header.
    - **Two ways out of the loop.** The loop ends when the values run out (`done`) or at the
      `break`, and both paths lead to `empty test`, which `loop` dominates.

    Returns
    -------
    Graph
        The control flow graph.

    Examples
    --------
    Draw the graph from `entry` to `exit`, with the branch on each jump:

    ```{python}
    import edgecase as ec

    cfg = ec.control_flow()

    cfg.plot(layout="pos", label_by="label", edge_label_by="branch")
    ```

    The code of each block:

    ```{python}
    cfg.get_node_df().select("label", "code")
    ```
    """
    nodes = [(name, name, {"code": code, "pos": (x, y)}) for name, (code, x, y) in _BLOCKS.items()]
    edges = [(u, v, None, {"branch": b} if b else {}) for u, v, b in _JUMPS]
    return _build_graph(nodes, edges, True, False, "Control flow", "control_flow()")
