import ast
import contextlib
import importlib.util
import re
import sys
import traceback
import webbrowser
from pathlib import Path
from typing import List, Tuple

import edgecase.viz


def extract_code_blocks_from_docstring(docstring: str) -> List[str]:
    """Extract Python code blocks from a docstring.

    Looks for code blocks marked with:
    - ```{python} ... ``` (Quarto, which the docs build executes; used by nearly every example)
    - ```python ... ```
    - ```pycon ... ``` (interactive Python)
    - >>> ... (doctest style)
    """
    code_blocks = []

    # Pattern 1: Fenced code blocks (```{python}, ```python or ```pycon). The Quarto form used to
    # be missed, so almost no docstring examples were actually run.
    fenced_pattern = r"```(?:\{python\}|python|pycon)[ \t]*\n(.*?)```"
    for match in re.finditer(fenced_pattern, docstring, re.DOTALL):
        code = match.group(1).strip()
        if code:
            code_blocks.append(code)

    # Pattern 2: Doctest style (>>> lines)
    doctest_lines = []
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.startswith(">>> "):
            doctest_lines.append(stripped[4:])
        elif stripped.startswith("... "):
            doctest_lines.append(stripped[4:])
        elif doctest_lines and stripped and not stripped.startswith("#"):
            # End of doctest block
            if doctest_lines:
                code_blocks.append("\n".join(doctest_lines))
                doctest_lines = []

    if doctest_lines:
        code_blocks.append("\n".join(doctest_lines))

    return code_blocks


def extract_docstrings_from_file(filepath: Path) -> List[Tuple[str, str, str]]:
    """Extract all docstrings from a Python file.

    Returns:
        List of (context, docstring, code_block) tuples where context identifies
        the location (module, class, function name).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source, filename=str(filepath))
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

    results = []

    def visit_node(node, context_parts: List[str]):
        docstring = ast.get_docstring(node)
        if docstring:
            code_blocks = extract_code_blocks_from_docstring(docstring)
            for code_block in code_blocks:
                context = ".".join(context_parts)
                results.append((context, docstring, code_block))

        # Visit class and function definitions
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_node(child, context_parts + [child.name])

    # Start with module-level docstring
    module_docstring = ast.get_docstring(tree)
    if module_docstring:
        code_blocks = extract_code_blocks_from_docstring(module_docstring)
        for code_block in code_blocks:
            results.append((filepath.stem, module_docstring, code_block))

    # Visit all top-level nodes
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_node(node, [filepath.stem, node.name])

    return results


@contextlib.contextmanager
def _no_browser():
    """Stop examples from opening browser tabs, restoring both functions afterwards.

    (These patches used to be applied inside every namespace and never undone, so each docstring
    group wrapped `plot_graph` once more and later tests in the same process saw the mocks.)
    """
    real_plot_graph = edgecase.viz.plot_graph
    real_open = webbrowser.open

    def _plot_graph_no_browser(*args, open_browser=True, **kwargs):
        return real_plot_graph(*args, open_browser=False, **kwargs)

    edgecase.viz.plot_graph = _plot_graph_no_browser
    webbrowser.open = lambda url, new=0, autoraise=True: None
    try:
        yield
    finally:
        edgecase.viz.plot_graph = real_plot_graph
        webbrowser.open = real_open


def _make_namespace() -> dict:
    """Create a fresh execution namespace with edgecase pre-imported."""
    namespace = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
    }
    try:
        exec("from edgecase import Graph", namespace)
        exec("import edgecase", namespace)
        exec("import edgecase as ec", namespace)
    except ImportError as e:
        raise RuntimeError(f"Failed to import edgecase: {e}") from e
    return namespace


def execute_code_example(
    code: str, context: str, namespace: dict | None = None
) -> Tuple[bool, str]:
    """Execute a code example and return (success, error_message).

    Args:
        code: The Python code to execute
        context: Context string for error reporting
        namespace: Optional shared namespace; a fresh one is created if not given.

    Returns:
        (True, "") if successful, (True, reason) if skipped for a missing optional package,
        (False, error_msg) if failed
    """
    if namespace is None:
        namespace = _make_namespace()

    try:
        exec(code, namespace)
        return True, ""
    except ImportError as e:
        # Examples for optional extras (plot_cosmo() needs cosmograph) can't run without the
        # package, so they count as skipped rather than failed
        missing = e.name or getattr(e.__cause__, "name", None)
        if missing and missing.split(".")[0] != "edgecase":
            return True, f"{missing} is not installed"
        tb = traceback.format_exc()
        return False, f"{type(e).__name__}: {e}\n{tb}"
    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{type(e).__name__}: {e}\n{tb}"


def test_all_docstring_examples():
    """Main test function that extracts and tests all docstring examples."""
    # Find all Python files in the edgecase package
    project_root = Path(__file__).parent.parent
    edgecase_dir = project_root / "python" / "edgecase"

    if not edgecase_dir.exists():
        print(f"Error: Could not find edgecase directory at {edgecase_dir}")
        sys.exit(1)

    # Collect all Python files
    python_files = list(edgecase_dir.rglob("*.py"))

    print(f"Found {len(python_files)} Python files to scan")
    print("=" * 80)

    all_examples = []

    # Extract all code examples
    for filepath in python_files:
        examples = extract_docstrings_from_file(filepath)
        for context, docstring, code in examples:
            all_examples.append((filepath, context, code))

    print(f"Found {len(all_examples)} code examples in docstrings")
    print("=" * 80)

    # Test each example, sharing namespace across all code blocks within the same
    # docstring so that later blocks can reference variables set in earlier ones.
    failures = []
    successes = 0
    skipped = 0
    shared_ns: dict = {}  # namespace shared within a (filepath, context) group
    current_group: tuple = ()

    with _no_browser():
        for i, (filepath, context, code) in enumerate(all_examples, 1):
            location = f"{filepath.relative_to(project_root)}: {context}"
            print(f"\n[{i}/{len(all_examples)}] Testing: {location}")
            print(f"Code preview: {code[:100]}...")

            group_key = (filepath, context)
            if group_key != current_group:
                shared_ns = _make_namespace()
                current_group = group_key

            success, error = execute_code_example(code, context, namespace=shared_ns)

            if success and error:
                print(f"- SKIPPED ({error})")
                skipped += 1
            elif success:
                print("✓ PASSED")
                successes += 1
            else:
                print("✗ FAILED")
                failures.append((location, code, error))

    # Print summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {successes}/{len(all_examples)} examples passed, {skipped} skipped")
    print("=" * 80)

    if failures:
        print(f"\n{len(failures)} FAILURES:\n")
        for i, (location, code, error) in enumerate(failures, 1):
            print(f"\n--- Failure {i}: {location} ---")
            print("Code:")
            print(code)
            print("\nError:")
            print(error)
            print("-" * 80)

        # Fail the test
        assert False, f"{len(failures)}/{len(all_examples)} docstring examples failed"
    else:
        print("\n✓ All docstring examples passed!")


def test_docstring_harness_restores_patches():
    """test_docstring_examples used to replace viz.plot_graph and webbrowser.open for the rest of
    the session (wrapping plot_graph once more per docstring group)."""
    before = (edgecase.viz.plot_graph, webbrowser.open)
    with _no_browser():
        assert edgecase.viz.plot_graph is not before[0]
    assert (edgecase.viz.plot_graph, webbrowser.open) == before


if __name__ == "__main__":
    test_all_docstring_examples()
