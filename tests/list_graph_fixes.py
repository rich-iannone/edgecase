"""
Script to generate specific line-by-line fixes for graph.py docstrings.
"""

import re

with open('/tmp/docstring_failures2.txt', 'r') as f:
    content = f.read()

# Find all graph.py failures
pattern = r'--- Failure \d+: python/edgecase/graph\.py: (graph\.Graph\.\w+) ---\nCode:\n(.*?)\n\nError:\n(.*?)\n-{70}'
matches = re.findall(pattern, content, re.DOTALL)

print(f"GRAPH.PY DOCSTRING FIXES NEEDED")
print("=" * 80)
print(f"Total: {len(matches)} failures\n")

# Group by method
from collections import defaultdict
by_method = defaultdict(list)

for method, code, error in matches:
    error_type = error.strip().split('\n')[0]
    by_method[method].append((error_type, code[:200]))

# Print organized list
for method in sorted(by_method.keys()):
    failures = by_method[method]
    print(f"\n{method} ({len(failures)} failures):")
    for error_type, code in failures:
        code_preview = code.strip().replace('\n', ' ')[:100]
        print(f"  - {error_type[:80]}")
        print(f"    Code: {code_preview}...")

print(f"\n\nTotal methods with failures: {len(by_method)}")
