"""
Test that all code examples in docstrings execute without errors.

This script extracts Python code blocks from docstrings across the codebase
and executes them to ensure documentation examples remain valid.
"""

import ast
import importlib.util
import re
import sys
import traceback
from pathlib import Path
from typing import List, Tuple


def extract_code_blocks_from_docstring(docstring: str) -> List[str]:
    """Extract Python code blocks from a docstring.

    Looks for code blocks marked with:
    - ```python ... ```
    - ```pycon ... ``` (interactive Python)
    - >>> ... (doctest style)
    """
    code_blocks = []

    # Pattern 1: Fenced code blocks (```python or ```pycon)
    fenced_pattern = r"```(?:python|pycon)\s*\n(.*?)```"
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


def execute_code_example(code: str, context: str) -> Tuple[bool, str]:
    """Execute a code example and return (success, error_message).

    Args:
        code: The Python code to execute
        context: Context string for error reporting

    Returns:
        (True, "") if successful, (False, error_msg) if failed
    """
    # Create a fresh namespace for each example
    namespace = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
    }

    # Import common modules that examples might use
    try:
        exec("from edgecase import Graph", namespace)
        exec("import edgecase", namespace)
        exec("import edgecase as ec", namespace)  # Common alias

        # Mock plot_graph to prevent browser windows from opening during tests
        exec(
            """
# Patch the plot_graph function to disable browser opening
import edgecase.viz
_original_plot_graph = edgecase.viz.plot_graph

def _mock_plot_graph(*args, open_browser=True, **kwargs):
    # Force browser opening to False during tests
    return _original_plot_graph(*args, open_browser=False, **kwargs)

edgecase.viz.plot_graph = _mock_plot_graph
""",
            namespace,
        )
    except ImportError as e:
        return False, f"Failed to import edgecase: {e}"

    try:
        exec(code, namespace)
        return True, ""
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

    # Test each example
    failures = []
    successes = 0

    for i, (filepath, context, code) in enumerate(all_examples, 1):
        location = f"{filepath.relative_to(project_root)}: {context}"
        print(f"\n[{i}/{len(all_examples)}] Testing: {location}")
        print(f"Code preview: {code[:100]}...")

        success, error = execute_code_example(code, context)

        if success:
            print("✓ PASSED")
            successes += 1
        else:
            print("✗ FAILED")
            failures.append((location, code, error))

    # Print summary
    print("\n" + "=" * 80)
    print(f"SUMMARY: {successes}/{len(all_examples)} examples passed")
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


if __name__ == "__main__":
    test_all_docstring_examples()
