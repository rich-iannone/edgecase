from edgecase import Graph


def demo_basic_dag():
    """Basic topological sort on a simple DAG."""
    print("=" * 60)
    print("Demo 1: Basic Topological Sort")
    print("=" * 60)

    g = Graph(directed=True)
    for node in ["A", "B", "C", "D"]:
        g.add_node(label=node)

    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D")

    print("Graph edges:")
    print("  A -> B, A -> C")
    print("  B -> D, C -> D")
    print("\nTopological order:", " → ".join(g.topological_sort()))


def demo_task_scheduling():
    """Use topological sort for task scheduling."""
    print("\n" + "=" * 60)
    print("Demo 2: Task Scheduling")
    print("=" * 60)

    g = Graph(directed=True)

    tasks = {
        "wake_up": [],
        "shower": ["wake_up"],
        "breakfast": ["wake_up"],
        "dress": ["shower"],
        "pack_bag": ["dress"],
        "leave_house": ["breakfast", "pack_bag"],
    }

    # Build dependency graph
    for task in tasks:
        g.add_node(label=task)

    for task, dependencies in tasks.items():
        for dep in dependencies:
            g.add_edge(dep, task)

    print("Morning routine with dependencies:")
    for task, deps in tasks.items():
        if deps:
            print(f"  {task}: requires {', '.join(deps)}")
        else:
            print(f"  {task}: no dependencies")

    order = g.topological_sort()
    print("\nOptimal task order:")
    for i, task in enumerate(order, 1):
        print(f"  {i}. {task.replace('_', ' ').title()}")


def demo_build_system():
    """Use topological sort for build system dependencies."""
    print("\n" + "=" * 60)
    print("Demo 3: Build System Dependencies")
    print("=" * 60)

    g = Graph(directed=True)

    files = {
        "config.h": [],
        "types.h": ["config.h"],
        "utils.h": ["types.h"],
        "utils.c": ["utils.h"],
        "main.h": ["types.h"],
        "main.c": ["main.h", "utils.h"],
    }

    for file in files:
        g.add_node(label=file)

    for file, deps in files.items():
        for dep in deps:
            g.add_edge(dep, file)

    print("Source file dependencies:")
    for file, deps in files.items():
        if deps:
            print(f"  {file}: includes {', '.join(deps)}")
        else:
            print(f"  {file}: no includes")

    build_order = g.topological_sort()
    print("\nBuild order (compilation sequence):")
    for i, file in enumerate(build_order, 1):
        print(f"  {i}. {file}")


def demo_course_prerequisites():
    """Use topological sort for course curriculum planning."""
    print("\n" + "=" * 60)
    print("Demo 4: University Course Prerequisites")
    print("=" * 60)

    g = Graph(directed=True)

    courses = {
        "CS101: Intro to Programming": [],
        "CS102: Data Structures": ["CS101: Intro to Programming"],
        "MATH101: Calculus I": [],
        "MATH102: Calculus II": ["MATH101: Calculus I"],
        "CS201: Algorithms": ["CS102: Data Structures", "MATH101: Calculus I"],
        "CS301: Advanced Algorithms": ["CS201: Algorithms", "MATH102: Calculus II"],
    }

    for course in courses:
        g.add_node(label=course)

    for course, prereqs in courses.items():
        for prereq in prereqs:
            g.add_edge(prereq, course)

    print("Course prerequisites:")
    for course, prereqs in courses.items():
        if prereqs:
            print(f"  {course}")
            for prereq in prereqs:
                print(f"    → requires: {prereq}")
        else:
            print(f"  {course}: no prerequisites")

    curriculum = g.topological_sort()
    print("\nSuggested course order:")
    for i, course in enumerate(curriculum, 1):
        semester = (i + 1) // 2  # 2 courses per semester
        print(f"  Semester {semester}: {course}")


def demo_cycle_detection():
    """Demonstrate cycle detection."""
    print("\n" + "=" * 60)
    print("Demo 5: Cycle Detection")
    print("=" * 60)

    # Valid DAG
    g1 = Graph(directed=True)
    for node in ["A", "B", "C"]:
        g1.add_node(label=node)
    g1.add_edge("A", "B")
    g1.add_edge("B", "C")

    print("Graph 1: A → B → C")
    print(f"  Is DAG: {g1.is_dag()}")
    print(f"  Topological order: {' → '.join(g1.topological_sort())}")

    # Cyclic graph
    g2 = Graph(directed=True)
    for node in ["X", "Y", "Z"]:
        g2.add_node(label=node)
    g2.add_edge("X", "Y")
    g2.add_edge("Y", "Z")
    g2.add_edge("Z", "X")  # Creates cycle

    print("\nGraph 2: X → Y → Z → X (cycle)")
    print(f"  Is DAG: {g2.is_dag()}")
    try:
        g2.topological_sort()
        print("  ERROR: Should have detected cycle!")
    except ValueError as e:
        print("  Topological sort failed: Cycle detected ✓")

    # Circular dependencies in packages
    g3 = Graph(directed=True)
    packages = ["pkg_A", "pkg_B", "pkg_C"]
    for pkg in packages:
        g3.add_node(label=pkg)
    g3.add_edge("pkg_A", "pkg_B")  # A depends on B
    g3.add_edge("pkg_B", "pkg_C")  # B depends on C
    g3.add_edge("pkg_C", "pkg_A")  # C depends on A - circular!

    print("\nPackage dependencies with circular reference:")
    print("  pkg_A requires pkg_B")
    print("  pkg_B requires pkg_C")
    print("  pkg_C requires pkg_A (circular!)")
    try:
        g3.topological_sort()
    except ValueError:
        print("  → Cannot resolve installation order (circular dependency)")


def demo_multiple_valid_orders():
    """Show that some DAGs have multiple valid topological orders."""
    print("\n" + "=" * 60)
    print("Demo 6: Multiple Valid Orderings")
    print("=" * 60)

    g = Graph(directed=True)
    for node in ["A", "B", "C", "D"]:
        g.add_node(label=node)

    # A and B can be done in any order, both feed into C and D
    g.add_edge("A", "C")
    g.add_edge("A", "D")
    g.add_edge("B", "C")
    g.add_edge("B", "D")

    print("Graph structure:")
    print("  A → C, A → D")
    print("  B → C, B → D")
    print("  (A and B are independent)")

    order = g.topological_sort()
    print(f"\nOne valid ordering: {' → '.join(order)}")
    print("\nNote: Other valid orderings exist, such as:")
    print("  - If starting with B instead of A")
    print("  - Different order of C and D at the end")


def demo_diamond_pattern():
    """Show diamond dependency pattern."""
    print("\n" + "=" * 60)
    print("Demo 7: Diamond Dependency Pattern")
    print("=" * 60)

    g = Graph(directed=True)
    for node in ["Database", "Cache", "API", "Frontend"]:
        g.add_node(label=node)

    g.add_edge("Database", "Cache")
    g.add_edge("Database", "API")
    g.add_edge("Cache", "Frontend")
    g.add_edge("API", "Frontend")

    print("Service dependency graph:")
    print("  Database → Cache → Frontend")
    print("  Database → API → Frontend")
    print("  (Diamond pattern)")

    startup_order = g.topological_sort()
    print("\nService startup order:")
    for i, service in enumerate(startup_order, 1):
        print(f"  {i}. Start {service}")


if __name__ == "__main__":
    demo_basic_dag()
    demo_task_scheduling()
    demo_build_system()
    demo_course_prerequisites()
    demo_cycle_detection()
    demo_multiple_valid_orders()
    demo_diamond_pattern()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
