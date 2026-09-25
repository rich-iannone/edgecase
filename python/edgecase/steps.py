from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class StepDelta:
    """What changed in a single step."""

    nodes_added: list[str] = field(default_factory=list)
    nodes_removed: list[str] = field(default_factory=list)
    edges_added: list[tuple[str, str]] = field(default_factory=list)
    edges_removed: list[tuple[str, str]] = field(default_factory=list)
    attrs_changed: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    """A single recorded operation on a Graph."""

    index: int
    kind: str
    method: str
    code: str
    narration: str = ""
    delta: StepDelta = field(default_factory=StepDelta)
    result: Any = None
    sub_steps: list[Step] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)

    def __repr__(self) -> str:
        sub = f"  [{len(self.sub_steps)} sub-steps]" if self.sub_steps else ""
        return f"Step({self.index}, {self.kind!r}, {self.method!r}{sub})"


class StepList(list):
    """A list of Steps with Polars-style filtering and export."""

    def filter(
        self,
        kind: Optional[str | list[str]] = None,
        method: Optional[str | list[str]] = None,
        involves: Optional[str | list[str]] = None,
        negate: bool = False,
    ) -> StepList:
        """Filter steps by kind, method, or involved nodes."""
        result = StepList()

        if isinstance(kind, str):
            kind = [kind]
        if isinstance(method, str):
            method = [method]
        if isinstance(involves, str):
            involves = [involves]

        for step in self:
            match = True

            if kind is not None:
                match = match and step.kind in kind
            if method is not None:
                match = match and step.method in method
            if involves is not None:
                touched = set(step.delta.nodes_added + step.delta.nodes_removed)
                for src, tgt in step.delta.edges_added + step.delta.edges_removed:
                    touched.add(src)
                    touched.add(tgt)
                for sub in step.sub_steps:
                    touched.update(sub.delta.nodes_added + sub.delta.nodes_removed)
                match = match and bool(touched & set(involves))

            if negate:
                match = not match

            if match:
                result.append(step)

        return result

    def head(self, n: int = 5) -> StepList:
        return StepList(self[:n])

    def tail(self, n: int = 5) -> StepList:
        return StepList(self[-n:])

    def select(self, *fields: str) -> list[dict]:
        """Extract specific fields from each step."""
        return [{f: getattr(s, f, None) for f in fields} for s in self]

    def to_dicts(self) -> list[dict]:
        """Export steps as JSON-serializable dicts."""
        results = []
        for s in self:
            d = {
                "index": s.index,
                "kind": s.kind,
                "method": s.method,
                "code": s.code,
                "narration": s.narration,
                "nodes_added": s.delta.nodes_added,
                "nodes_removed": s.delta.nodes_removed,
                "edges_added": s.delta.edges_added,
                "edges_removed": s.delta.edges_removed,
                "sub_step_count": len(s.sub_steps),
                "timestamp": s.timestamp,
            }
            results.append(d)
        return results

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dicts(), indent=2)

    def to_polars(self):
        import polars as pl

        return pl.DataFrame(self.to_dicts())

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.to_dicts())

    def __repr__(self) -> str:
        if not self:
            return "StepList([])"
        lines = []
        for s in self:
            sub = f"  [{len(s.sub_steps)} sub-steps]" if s.sub_steps else ""
            lines.append(f"  {s.index + 1:>3}  {s.kind:<14s} {s.method}(){sub}")
        return f"StepList([\n" + "\n".join(lines) + "\n])"

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return StepList(result)
        return result


# ── Step kind constants ─────────────────────────────────────────────────

KIND_BUILD_NODES = "build_nodes"
KIND_BUILD_EDGES = "build_edges"
KIND_REMOVE = "remove"
KIND_MUTATE = "mutate"
KIND_QUERY = "query"
KIND_METRIC = "metric"
KIND_TRAVERSE = "traverse"
KIND_SELECT = "select"
KIND_TRANSFORM = "transform"
KIND_ANNOTATE = "annotate"
KIND_GENERATE = "generate"


# ── Narration templates ─────────────────────────────────────────────────


def auto_narrate(step: Step) -> str:
    """Generate a human-readable narration for a step."""
    d = step.delta

    if step.kind == KIND_ANNOTATE:
        return step.narration

    if step.kind == KIND_BUILD_NODES:
        n = len(d.nodes_added)
        if n == 1:
            return f"Node '{d.nodes_added[0]}' added."
        labels = ", ".join(d.nodes_added[:8])
        suffix = f" and {n - 8} more" if n > 8 else ""
        return f"{n} nodes added: {labels}{suffix}."

    if step.kind == KIND_BUILD_EDGES:
        n = len(d.edges_added)
        if n == 1:
            src, tgt = d.edges_added[0]
            return f"Edge {src} → {tgt} added."
        nodes = set()
        for src, tgt in d.edges_added:
            nodes.add(src)
            nodes.add(tgt)
        return f"{n} edges added, connecting {len(nodes)} nodes."

    if step.kind == KIND_REMOVE:
        parts = []
        if d.nodes_removed:
            parts.append(f"{len(d.nodes_removed)} node(s) removed")
        if d.edges_removed:
            parts.append(f"{len(d.edges_removed)} edge(s) removed")
        return "; ".join(parts) + "."

    if step.kind == KIND_TRAVERSE:
        n = len(step.sub_steps)
        return f"{step.method.upper()} traversal: {n} nodes visited."

    if step.kind == KIND_GENERATE:
        parts = []
        if d.nodes_added:
            parts.append(f"{len(d.nodes_added)} nodes")
        if d.edges_added:
            parts.append(f"{len(d.edges_added)} edges")
        detail = " and ".join(parts)
        if detail:
            return f"{step.method}() added {detail}."
        return f"Graph structure generated via {step.method}()."

    return f"{step.method}() called."
