import re
from collections import defaultdict

# Read the failures file
with open("/tmp/docstring_failures.txt", "r") as f:
    content = f.read()

# Extract failures
failure_pattern = r"--- Failure \d+: (.*?) ---\nCode:\n(.*?)\n\nError:\n(.*?)\n-{70}"
failures = re.findall(failure_pattern, content, re.DOTALL)

print(f"Total failures: {len(failures)}\n")

# Categorize by error type
error_types = defaultdict(list)
for location, code, error in failures:
    # Get the main error message
    error_lines = error.strip().split("\n")
    if error_lines:
        main_error = error_lines[0]
        error_types[main_error].append((location, code[:100]))

# Print categorized errors
print("=" * 80)
print("ERRORS BY TYPE:")
print("=" * 80)

for error_msg, examples in sorted(error_types.items(), key=lambda x: -len(x[1])):
    print(f"\n{len(examples)} failures: {error_msg}")
    print("-" * 80)
    for location, code_preview in examples[:3]:  # Show first 3 examples
        print(f"  Location: {location}")
        print(f"  Code: {code_preview}...")
    if len(examples) > 3:
        print(f"  ... and {len(examples) - 3} more")
