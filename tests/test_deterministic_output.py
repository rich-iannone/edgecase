"""Output that must not change from one Python process to the next.

Rust's standard `HashMap` is seeded randomly per process, and Python's string hashing depends on
`PYTHONHASHSEED`, so anything built by iterating either one can come out in a different order on
every run (e.g., between two renders of the docs). Comparing within one process can't catch that,
so these tests run the same code in separate processes.
"""

import os
import subprocess
import sys

import edgecase as ec

_DICTS = """
import edgecase as ec
g = ec.Graph().add_nodes(list("ABCDE")).add_edges([("A", "B"), ("A", "C"), ("B", "D"), ("D", "E")])
h = ec.Graph().add_nodes(list("ABCDE")).add_edges([("B", "A"), ("B", "D"), ("C", "D"), ("C", "E")])
a, b, d, e = (g.get_id_from_label(x) for x in "ABDE")
m = ec.Graph(directed=True).add_nodes(list("ABC"))
m.add_edges([("C", "C"), ("A", "A"), ("B", "B"), ("C", "B"), ("C", "B"), ("A", "B"), ("A", "B")])
from edgecase._core import force_directed_layout_rust
print(g.edge_overlap(h))
print(g.differential_edges(h))
print(g.graph_edit_distance(h))
for method in ["closest", "shortest", "kernel", "separation"]:
    print(g.network_proximity([a, b], [d, e], method=method))
print(m.count_loops(), m.count_multiple())
print(g.label_propagation({e: 2, a: 1}, return_dict=True))
print(list(force_directed_layout_rust(20, [(i, i + 1) for i in range(19)], seed=1)))
"""

_PLAYERS = """
import re
import edgecase as ec
g = ec.karate_club()
rec = g.record_traversal()
rec.select_nodes_by_id([1]).trav_out().trav_out_edge()
h = ec.Graph().add_nodes(["Ana", "Ben", "Cy", "Dee"]).add_edges([("Ana", "Ben"), ("Ben", "Cy")])
h.add_edge("Cy", "Dee").bfs(source="Ana")
for player in [
    g.plot_traversal(source=1, seed=1),
    rec.plot(seed=1),
    h.play(seed=1),
    h.play(layout_mode="stable", seed=1),
]:
    print(player._repr_html_())
"""


def _run(code: str, hash_seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    return out.stdout


def test_rust_dicts_have_a_fixed_key_order():
    """Regression: dicts built from a Rust `std::collections::HashMap` had a random key order."""
    runs = {_run(_DICTS, seed) for seed in ("0", "1", "2")}
    assert len(runs) == 1


def test_rust_dict_key_order():
    """Result dicts list their fields in the documented order, and ID-keyed ones by ID."""
    g = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("B", "C")])
    h = ec.Graph().add_nodes(list("ABCD")).add_edges([("A", "B"), ("C", "D")])
    assert list(g.edge_overlap(h)) == ["jaccard", "overlap", "union_size", "g1_only", "g2_only"]
    assert list(g.differential_edges(h)) == ["added", "removed"]
    assert list(g.graph_edit_distance(h)) == [
        "distance",
        "node_insertions",
        "node_deletions",
        "edge_insertions",
        "edge_deletions",
    ]
    a, d = g.get_id_from_label("A"), g.get_id_from_label("D")
    for method in ["closest", "shortest", "kernel"]:
        result = g.network_proximity([a], [d], method=method)
        assert list(result) == ["proximity", "reachable_pairs", "total_pairs"]
    assert list(g.network_proximity([a], [d], method="separation")) == [
        "separation",
        "d_ab",
        "d_aa",
        "d_bb",
    ]

    loops = ec.Graph().add_nodes(list("ABCD")).add_edges([("D", "D"), ("B", "B"), ("C", "C")])
    assert list(loops.count_loops()) == sorted(loops.count_loops())
    multi = ec.Graph().add_nodes(list("ABC")).add_edges([("C", "B")] * 2 + [("A", "B")] * 2)
    assert list(multi.count_multiple()) == sorted(multi.count_multiple())
    labels = g.label_propagation({d: 2, a: 1}, return_dict=True)
    assert list(labels) == sorted(labels)


def test_differential_edges_are_sorted():
    g = ec.Graph().add_nodes(list("ABCDE")).add_edges([("D", "E"), ("A", "C"), ("B", "D")])
    h = ec.Graph().add_nodes(list("ABCDE")).add_edges([("C", "E"), ("B", "D"), ("C", "D")])
    assert g.differential_edges(h) == {
        "added": [("C", "D"), ("C", "E")],
        "removed": [("A", "C"), ("D", "E")],
    }


def test_seeded_step_players_ignore_hash_seed():
    """Regression: plot_traversal() and the recorder serialized Python sets of node IDs, so their
    HTML changed with `PYTHONHASHSEED` even when seeded."""
    runs = {_run(_PLAYERS, seed) for seed in ("0", "1", "2")}
    assert len(runs) == 1
