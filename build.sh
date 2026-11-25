#!/usr/bin/env bash
# Usage: ./build.sh [debug|release] [--clean] [--test] [--python]
#   --clean   wipe build/ first
#   --test    run C++ tests after building
#   --python  create .venv and install python/requirements.txt
set -euo pipefail
cd "$(dirname "$0")"

BUILD_TYPE=Release
CLEAN=0; TEST=0; PY=0
for arg in "$@"; do
  case "$arg" in
    debug)    BUILD_TYPE=Debug ;;
    release)  BUILD_TYPE=Release ;;
    --clean)  CLEAN=1 ;;
    --test)   TEST=1 ;;
    --python) PY=1 ;;
    *) echo "unknown arg: $arg"; exit 1 ;;
  esac
done

[[ $CLEAN -eq 1 ]] && rm -rf build

# If the project folder was moved/renamed, the CMake cache points at the old
# path and CMake refuses to reuse it -- wipe it and reconfigure.
if [[ -f build/CMakeCache.txt ]] && \
   ! grep -qx "CMAKE_HOME_DIRECTORY:INTERNAL=$PWD" build/CMakeCache.txt; then
  echo "build.sh: project moved, removing stale build/"
  rm -rf build
fi

# macOS: if the default SDK can't link (toolchain/SDK mismatch), fall back to
# the newest installed SDK that works. Override by setting SDKROOT yourself.
if [[ "$(uname)" == "Darwin" && -z "${SDKROOT:-}" ]]; then
  tmp="$(mktemp -d)"; echo 'int main(){}' > "$tmp/t.cpp"
  if ! c++ "$tmp/t.cpp" -o "$tmp/t" 2>/dev/null; then
    for sdk in $(ls -d /Library/Developer/CommandLineTools/SDKs/MacOSX[0-9]*.sdk 2>/dev/null | sort -rV); do
      if SDKROOT="$sdk" c++ "$tmp/t.cpp" -o "$tmp/t" 2>/dev/null; then
        export SDKROOT="$sdk"; echo "build.sh: using SDKROOT=$sdk"; break
      fi
    done
  fi
  rm -rf "$tmp"
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ${SDKROOT:+-DCMAKE_OSX_SYSROOT="$SDKROOT"}
cmake --build build -j

[[ $TEST -eq 1 ]] && ctest --test-dir build --output-on-failure

if [[ $PY -eq 1 ]]; then
  [[ -d .venv ]] || python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r python/requirements.txt
fi
