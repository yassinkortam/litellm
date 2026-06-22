#!/bin/sh
# Entrypoint for the free-threaded (no-GIL) LiteLLM proxy build.
#
# Sets PYTHON_GIL=0 (defense in depth — the Dockerfile already sets it) and
# logs whether the GIL is actually disabled at runtime. If a C extension
# imported by litellm forces the GIL back on, the warning will appear here
# and the benchmark numbers will look identical to the GIL-enabled build.

set -e

export PYTHON_GIL="${PYTHON_GIL:-0}"

python - <<'PY'
import sys
gil_check = getattr(sys, "_is_gil_enabled", None)
if gil_check is None:
    print("[freethreaded_entrypoint] WARNING: this interpreter does not "
          "expose sys._is_gil_enabled — not a free-threaded build.")
else:
    state = "ENABLED" if gil_check() else "DISABLED"
    print(f"[freethreaded_entrypoint] sys._is_gil_enabled() -> {state}")
    print(f"[freethreaded_entrypoint] sys.version: {sys.version}")
PY

if [ "$USE_DDTRACE" = "true" ]; then
    export DD_TRACE_OPENAI_ENABLED="False"
    exec ddtrace-run litellm "$@"
else
    exec litellm "$@"
fi
