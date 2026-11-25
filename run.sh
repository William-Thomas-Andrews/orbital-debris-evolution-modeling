#!/usr/bin/env bash
# Usage:
#   ./run.sh [args...]                 run the C++ executable
#   ./run.sh cpp [args...]             same as above
#   ./run.sh py <script.py> [args...]  run a script from python/scripts/ in .venv
set -euo pipefail
cd "$(dirname "$0")"

mode="${1:-cpp}"
[[ $# -gt 0 ]] && shift

case "$mode" in
  py)
    script="${1:?need a script name, e.g. run_baseline.py}"; shift
    PYTHONPATH="python${PYTHONPATH:+:$PYTHONPATH}" .venv/bin/python "python/scripts/$script" "$@"
    ;;
  cpp)
    [[ -x build/bin/mocat6rd ]] || ./build.sh
    build/bin/mocat6rd "$@"
    ;;
  *)
    [[ -x build/bin/mocat6rd ]] || ./build.sh
    build/bin/mocat6rd "$mode" "$@"
    ;;
esac
