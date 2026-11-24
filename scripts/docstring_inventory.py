"""Inventory of the Graph method docstrings, and progress against the target levels.

Reads `python/edgecase/graph.py` and the reference sections in `great-docs.yml`, records what
each public method's Examples section contains, and compares the level it reaches with the
target in `docs_plan/levels.csv` (see DOCSTRING_EXAMPLES_PLAN.md, §3 and §11).

Usage (from the repository root):

    python scripts/docstring_inventory.py              # per-family progress table (Markdown)
    python scripts/docstring_inventory.py --below      # methods below target, by family
    python scripts/docstring_inventory.py --below "Centrality"
    python scripts/docstring_inventory.py --json out.json   # the full per-method inventory

The current level is a heuristic: L0 has an Examples section; L1 adds a second block and some
prose (at least 40 words); L2 adds a `plot(` or a `play(`; L3 has both, or a second plot.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "python" / "edgecase" / "graph.py"
CONFIG = ROOT / "great-docs.yml"
LEVELS = ROOT / "docs_plan" / "levels.csv"

LEVEL_ORDER = ["-", "L0", "L1", "L2", "L3"]

# The reference sections each phase of the plan (§10) works through.
PHASES = {
    0: ["Visualization", "Step Log"],
    1: [
        "Centrality", "Network Propagation", "Community Detection", "Distance and Paths",
        "Connectivity and Flow", "Bridges and Connectivity", "Connected Components",
        "Spanning Trees", "Core Decomposition", "Graph Coloring", "Matching and Covering",
        "Graph Selection", "Querying", "Graph Traversal",
    ],
    2: [
        "Clustering", "Triangle and Transitivity", "Clique Operations", "Network Motif Census",
        "Directed Acyclic Graphs", "Path Analysis", "Cycle Detection", "Eulerian Paths",
        "Similarity", "Topological Overlap", "Link Prediction", "Assortativity",
        "Spectral Analysis", "Network Efficiency", "Distance Metrics", "Independent Sets",
        "Dominating Sets", "Approximation and Optimization", "Reciprocity", "Bipartite Analysis",
    ],
    3: [
        "Label Propagation", "Network Smoothing", "Seed Module Discovery", "Network Enrichment",
        "Network Proximity", "Network Robustness", "Influence Maximization", "WGCNA",
        "GWAS Pipeline", "Isomorphism and Graph Hashing", "Network Comparison", "Link Analysis",
    ],
}

DATASET_RE = re.compile(
    r"ec\.(karate_club|les_miserables|florentine_families|davis_southern_women|dolphins|football"
    r"|polbooks|jazz|celegans|power_grid|load_dataset)\("
)


def reference_sections() -> dict[str, str]:
    cfg = yaml.safe_load(CONFIG.read_text())
    section_of: dict[str, str] = {}
    for sec in cfg["reference"]["sections"]:
        for item in sec.get("contents", []):
            name = item if isinstance(item, str) else (item.get("name") or "")
            if name.startswith("Graph."):
                section_of.setdefault(name[6:], sec["title"])
    return section_of


def current_level(row: dict) -> str:
    if not row["has_examples"]:
        return "-"
    level = "L0"
    if row["n_blocks"] >= 2 and row["example_prose_words"] >= 40:
        level = "L1"
        # A step player counts as a plot, so a Step Log page can reach L2 with play() alone
        n_visuals = row["n_plots"] + row["uses_play"]
        if n_visuals >= 1:
            level = "L2"
            if n_visuals >= 2:
                level = "L3"
    return level


def inventory() -> list[dict]:
    src = GRAPH.read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Graph")
    section_of = reference_sections()

    rows = []
    for node in cls.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        is_property = any(
            isinstance(d, ast.Name) and d.id == "property" for d in node.decorator_list
        )
        doc = ast.get_docstring(node) or ""
        m = re.search(r"^Examples\n-+\n(.*)", doc, re.S | re.M)
        ex = m.group(1) if m else ""
        ex = re.split(r"^(?:See Also|Notes|References)\n-+\n", ex, flags=re.M)[0]
        blocks = re.findall(r"```\{python\}\n(.*?)```", ex, re.S)
        prose = re.sub(r"```\{python\}.*?```", "", ex, flags=re.S)
        code = "\n".join(blocks)
        row = {
            "name": node.name,
            "kind": "property" if is_property else "method",
            "section": section_of.get(node.name, "(not in reference)"),
            "lineno": node.lineno,
            "doc_lines": len(doc.splitlines()),
            "has_figure": "assets/graphs/" in doc,
            "has_examples": bool(m),
            "n_blocks": len(blocks),
            "example_prose_words": len(prose.split()),
            "n_plots": len(re.findall(r"\.plot(?:_vis|_cosmo|_traversal)?\(", code)),
            "uses_plot": ".plot(" in code,
            "uses_play": ".play(" in code,
            "uses_dataset": bool(DATASET_RE.search(code)),
            "uses_dataset_any": bool(re.search(r"ec\.[a-z_]+\(", code)),
            "uses_return_dict": "return_dict=True" in code,
            "uses_get_df": bool(re.search(r"get_(node|edge)_df\(", code)),
            "uses_table": bool(re.search(r"\.(tbl_|gt_|summary_table|centrality_table)", code)),
            "has_see_also": "See Also\n--------" in doc,
            "has_notes": "\nNotes\n-----" in doc,
            "has_raises": "\nRaises\n------" in doc,
        }
        row["level"] = current_level(row)
        rows.append(row)
    return rows


def read_targets() -> dict[str, dict]:
    if not LEVELS.exists():
        return {}
    with LEVELS.open(newline="") as fh:
        return {r["method"]: r for r in csv.DictReader(fh)}


def attach_targets(rows: list[dict], targets: dict[str, dict]) -> None:
    for row in rows:
        t = targets.get(row["name"], {})
        row["target"] = t.get("level", "")
        row["scenario"] = t.get("scenario", "")
        row["needs"] = t.get("needs", "")
        row["at_target"] = bool(row["target"]) and (
            LEVEL_ORDER.index(row["level"]) >= LEVEL_ORDER.index(row["target"])
        )


def phase_of(section: str) -> int:
    for phase, sections in PHASES.items():
        if section in sections:
            return phase
    return 4


def family_table(rows: list[dict]) -> str:
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["target"]:
            by_section[row["section"]].append(row)

    def order(section: str):
        phase = phase_of(section)
        listed = PHASES.get(phase, [])
        return (phase, listed.index(section) if section in listed else 0, section)

    lines = [
        "| Phase | Family | Methods | Target L3 / L2 / L0–L1 | Now L3 / L2 / L1 / L0 | At target |",
        "|---:|---|---:|---|---|---:|",
    ]
    totals = defaultdict(int)
    for section in sorted(by_section, key=order):
        group = by_section[section]
        tgt = {lv: sum(r["target"] == lv for r in group) for lv in LEVEL_ORDER}
        now = {lv: sum(r["level"] == lv for r in group) for lv in LEVEL_ORDER}
        done = sum(r["at_target"] for r in group)
        for lv in LEVEL_ORDER:
            totals["t" + lv] += tgt[lv]
            totals["n" + lv] += now[lv]
        totals["methods"] += len(group)
        totals["done"] += done
        lines.append(
            f"| {phase_of(section)} | {section} | {len(group)} | "
            f"{tgt['L3']} / {tgt['L2']} / {tgt['L1'] + tgt['L0']} | "
            f"{now['L3']} / {now['L2']} / {now['L1']} / {now['L0']} | "
            f"{done} ({100 * done // len(group)}%) |"
        )
    n = totals["methods"]
    lines.append(
        f"| | **Total** | **{n}** | {totals['tL3']} / {totals['tL2']} / "
        f"{totals['tL1'] + totals['tL0']} | {totals['nL3']} / {totals['nL2']} / {totals['nL1']} / "
        f"{totals['nL0']} | **{totals['done']} ({100 * totals['done'] // max(n, 1)}%)** |"
    )
    return "\n".join(lines)


def below_target(rows: list[dict], section: str | None) -> str:
    out = []
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["target"] and not row["at_target"]:
            if section is None or row["section"] == section:
                by_section[row["section"]].append(row)
    for sec in sorted(by_section, key=lambda s: (phase_of(s), s)):
        out.append(f"### {sec}\n")
        out.append("| Method | Now | Target | Scenario | Needs |")
        out.append("|---|---|---|---|---|")
        for r in by_section[sec]:
            out.append(
                f"| `{r['name']}` | {r['level']} | {r['target']} | {r['scenario']} | {r['needs']} |"
            )
        out.append("")
    return "\n".join(out) or "Every method is at its target level."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--below", nargs="?", const="", metavar="FAMILY",
                        help="list methods below their target level (optionally for one family)")
    parser.add_argument("--json", metavar="PATH", help="write the per-method inventory as JSON")
    args = parser.parse_args(argv)

    rows = inventory()
    targets = read_targets()
    attach_targets(rows, targets)

    untargeted = [r["name"] for r in rows if r["section"] != "(not in reference)" and not r["target"]]
    if untargeted:
        print(f"warning: no target level for {len(untargeted)} methods: "
              f"{', '.join(untargeted[:10])}", file=sys.stderr)
    stale = sorted(set(targets) - {r["name"] for r in rows})
    if stale:
        print(f"warning: levels.csv names {len(stale)} unknown methods: {', '.join(stale[:10])}",
              file=sys.stderr)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    if args.below is not None:
        print(below_target(rows, args.below or None))
    elif not args.json:
        print(family_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
