#!/bin/sh
# Keep the Conda and PyTorch OpenMP consumers on one library without overriding BLAS.
set -eu
revision_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
revision_python="$revision_root/.sim-env/bin/python"
revision_omp="$revision_root/.sim-env/lib/libomp.dylib"
if [ "$(uname -s)" = Darwin ] && [ -f "$revision_omp" ]; then
    revision_lib="$revision_root/artifacts/revision_runtime/lib"
    mkdir -p "$revision_lib"
    ln -sf "$revision_omp" "$revision_lib/libomp.dylib"
    exec env DYLD_LIBRARY_PATH="$revision_lib" "$revision_python" "$@"
fi
exec "$revision_python" "$@"
