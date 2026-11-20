#!/usr/bin/env bash
#
# Run NetworkX's own test suite against the edgecase backend.
#
# NetworkX's test-mode dispatching (NETWORKX_TEST_BACKEND=edgecase) converts the
# NetworkX graphs that tests build into edgecase graphs and calls edgecase's
# implementation of each dispatchable function the backend provides, then converts
# graph results back so the tests can check them.
#
# Usage:
#   scripts/run_nx_tests.sh [pytest args...]      # e.g. -n 8, -k pagerank
#   NX_TEST_MODULES="networkx.algorithms.components" scripts/run_nx_tests.sh
#
# Environment variables:
#   NETWORKX_TEST_BACKEND   default "edgecase"; set to "" for a plain NetworkX run
#   NETWORKX_FALLBACK_TO_NX default "True": call NetworkX itself for functions
#                           edgecase does not implement (otherwise those xfail)
#   NX_TEST_MODULES         default "networkx": the test packages/modules to run
#   EDGECASE_NX_ROUNDTRIP   "1" makes the backend claim every dispatchable function
#                           that takes graphs and does not mutate them, convert its
#                           graphs to edgecase and back, and run the NetworkX code on
#                           the result: a test of conversion fidelity across the suite
#
# The run uses NetworkX's warning filters (warnings are errors) and none of
# edgecase's pytest configuration, and does not write a cache.
set -euo pipefail

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

export NETWORKX_TEST_BACKEND="${NETWORKX_TEST_BACKEND-edgecase}"
export NETWORKX_FALLBACK_TO_NX="${NETWORKX_FALLBACK_TO_NX-True}"
if [ -z "$NETWORKX_TEST_BACKEND" ]; then
    unset NETWORKX_TEST_BACKEND NETWORKX_FALLBACK_TO_NX
fi

# shellcheck disable=SC2086
python -m pytest \
    --pyargs ${NX_TEST_MODULES:-networkx} \
    -c /dev/null \
    --rootdir "$workdir" \
    -o filterwarnings=error \
    -p no:cacheprovider \
    "$@"
