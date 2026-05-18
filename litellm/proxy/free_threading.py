"""
Free-threading (no-GIL) support helpers for the LiteLLM proxy.

Python 3.13+ ships an optional free-threaded build (PEP 703) where the GIL can
be disabled, allowing multiple threads to execute Python bytecode in parallel.
On such a build the proxy can serve traffic from N threads inside a single
process instead of N separate uvicorn worker processes, sharing memory and
avoiding inter-process overhead.

See: https://docs.python.org/3/howto/free-threading-python.html

These helpers only *detect* the runtime; they never change interpreter state.
Free-threaded serving is strictly opt-in via the ``--run_free_threading`` CLI
flag (or ``LITELLM_RUN_FREE_THREADING=true``).
"""

import sys
import sysconfig
from typing import Optional


def is_gil_disabled_build() -> bool:
    """True if the interpreter was built with free-threading support.

    A free-threaded build sets the ``Py_GIL_DISABLED`` config var to 1. This is
    independent of whether the GIL is *currently* active (an extension module or
    ``PYTHON_GIL=1`` can re-enable it at runtime).
    """
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def is_gil_enabled() -> Optional[bool]:
    """Whether the GIL is currently active.

    Returns ``None`` on interpreters that predate ``sys._is_gil_enabled``
    (added in 3.13), where the GIL is always present.
    """
    is_gil_enabled_fn = getattr(sys, "_is_gil_enabled", None)
    if is_gil_enabled_fn is None:
        return None
    return bool(is_gil_enabled_fn())


def is_free_threaded() -> bool:
    """True only if this is a free-threaded build *and* the GIL is disabled.

    This is the condition under which running multiple server threads yields
    true CPU parallelism.
    """
    if not is_gil_disabled_build():
        return False
    return is_gil_enabled() is False


def free_threading_status() -> str:
    """Human-readable one-line summary of the free-threading runtime state."""
    version = sys.version.split()[0]
    if is_free_threaded():
        return (
            f"\033[1;32mLiteLLM Proxy: free-threaded Python {version} detected "
            f"(GIL disabled) — server threads run with true parallelism.\033[0m"
        )
    if is_gil_disabled_build():
        return (
            f"\033[1;33mLiteLLM Proxy: free-threaded Python {version} build "
            f"detected but the GIL is currently ENABLED (likely PYTHON_GIL=1 or "
            f"a GIL-requiring extension). Run with PYTHON_GIL=0 for true "
            f"parallelism.\033[0m"
        )
    return (
        f"\033[1;33mLiteLLM Proxy: Python {version} is a standard (GIL) build. "
        f"--run_free_threading will still run {{n}} server threads, but the GIL "
        f"serializes Python execution so this is mostly useful for parity "
        f"testing. Install free-threaded Python 3.13+ (e.g. python3.13t) for "
        f"real gains.\033[0m"
    )
