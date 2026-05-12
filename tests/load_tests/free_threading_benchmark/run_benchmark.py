"""
Orchestrator for the free-threading benchmark.

Spawns the mock upstream, spawns one or two litellm proxy processes
(GIL-on and GIL-off), runs the load generator against each, samples CPU
out-of-band, and emits a side-by-side report.

Usage:
    # compare both GIL states (requires python3.13t on PATH for the off run)
    python run_benchmark.py --mode compare --duration 60 --concurrency 64

    # single run, current interpreter
    python run_benchmark.py --mode single --gil on --duration 60

The script does NOT require root or Docker — everything runs as the
invoking user, on localhost. The mock upstream means there is zero
network egress and zero cost.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def _wait_for_port(host: str, port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as e:
            last_err = e
            time.sleep(0.25)
    raise RuntimeError(f"port {host}:{port} did not open in {timeout_s}s ({last_err})")


def _free_threaded_python() -> Optional[str]:
    """Return path to a python3.13t (or newer free-threaded) interpreter, or None."""
    for candidate in ("python3.13t", "python3.14t", "python3t"):
        path = shutil.which(candidate)
        if path:
            return path
    # Fall back to current interpreter if it's already free-threaded.
    if not getattr(sys, "_is_gil_enabled", lambda: True)():
        return sys.executable
    return None


def _spawn_mock_upstream(port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "tests.load_tests.free_threading_benchmark.mock_upstream",
        "--port",
        str(port),
        "--tokens",
        "64",
        "--first-token-ms",
        "40",
        "--inter-token-ms",
        "8",
    ]
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    _wait_for_port("127.0.0.1", port)
    return proc


def _spawn_proxy(
    *,
    python_bin: str,
    port: int,
    gil: str,
    config: Path,
    log_path: Path,
    upstream_url: str,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["MOCK_UPSTREAM_URL"] = upstream_url
    if gil == "off":
        env["PYTHON_GIL"] = "0"
    else:
        env.pop("PYTHON_GIL", None)
    cmd = [
        python_bin,
        "-m",
        "litellm",
        "--config",
        str(config),
        "--port",
        str(port),
        "--num_workers",
        "1",
        "--telemetry",
        "False",
    ]
    log = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
    )
    try:
        _wait_for_port("127.0.0.1", port, timeout_s=60.0)
    except Exception:
        proc.terminate()
        raise
    return proc


def _verify_gil_state(python_bin: str, want_off: bool) -> str:
    """Returns a short note string about the actual GIL state of python_bin."""
    out = subprocess.run(
        [
            python_bin,
            "-c",
            "import sys; f=getattr(sys,'_is_gil_enabled',None); "
            "print('gil_disabled' if (f and not f()) else 'gil_enabled')",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if want_off and out != "gil_disabled":
        return f"WARNING: requested GIL=off but interpreter reports {out}"
    if not want_off and out == "gil_disabled":
        return f"NOTE: requested GIL=on but interpreter is free-threaded ({out})"
    return f"interpreter reports {out}"


def _run_loadgen(
    *, label: str, port: int, duration: float, concurrency: int, out_dir: Path
) -> Path:
    out_path = out_dir / f"{label}.loadgen.json"
    cmd = [
        sys.executable,
        "-m",
        "tests.load_tests.free_threading_benchmark.loadgen",
        "--label",
        label,
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--api-key",
        "sk-bench",
        "--model",
        "mock-gpt",
        "--concurrency",
        str(concurrency),
        "--duration",
        str(duration),
        "--warmup",
        "5",
        "--out",
        str(out_path),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return out_path


def _terminate(proc: Optional[subprocess.Popen]) -> None:
    if not proc or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def run_one(
    *,
    label: str,
    python_bin: str,
    gil: str,
    proxy_port: int,
    upstream_port: int,
    duration: float,
    concurrency: int,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    note = _verify_gil_state(python_bin, want_off=(gil == "off"))
    print(f"[{label}] {note}")

    mock_proc = _spawn_mock_upstream(upstream_port)
    proxy_log = out_dir / f"{label}.proxy.log"
    proxy_proc = _spawn_proxy(
        python_bin=python_bin,
        port=proxy_port,
        gil=gil,
        config=HERE / "proxy_config.yaml",
        log_path=proxy_log,
        upstream_url=f"http://127.0.0.1:{upstream_port}/v1",
    )

    try:
        # Sample CPU concurrently with the load run.
        # We start the sampler slightly inside the duration window so it
        # doesn't catch the warmup ramp.
        cpu_pid = proxy_proc.pid
        cpu_wall = max(10.0, duration - 5.0)

        # Kick off loadgen first (it does its own warmup), then sample CPU
        # for cpu_wall seconds in a background subprocess.
        cpu_proc = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import json,sys;"
                "from tests.load_tests.free_threading_benchmark.cpu_sampler import sample_window;"
                f"w=sample_window({cpu_pid},{cpu_wall});"
                "sys.stdout.write(json.dumps({'pid':w.pid,'cpu_seconds':w.cpu_seconds,"
                "'wall_seconds':w.wall_seconds,'avg_cores':w.avg_cores}))",
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        loadgen_path = _run_loadgen(
            label=label,
            port=proxy_port,
            duration=duration,
            concurrency=concurrency,
            out_dir=out_dir,
        )
        cpu_out, _ = cpu_proc.communicate(timeout=duration + 30)
        cpu = json.loads(cpu_out.strip().splitlines()[-1])
    finally:
        _terminate(proxy_proc)
        _terminate(mock_proc)

    with open(loadgen_path) as f:
        result = json.load(f)
    result["proxy_pid"] = cpu["pid"]
    result["proxy_cpu_seconds"] = cpu["cpu_seconds"]
    result["proxy_cpu_wall_seconds"] = cpu["wall_seconds"]
    result["proxy_avg_cores"] = cpu["avg_cores"]
    if result["completed"] > 0:
        # CPU sampling window != loadgen window exactly; this is a per-request
        # average computed against the sampled window.
        per_req_ms = (
            (cpu["cpu_seconds"] / result["completed"])
            * 1000.0
            * (result["duration_s"] / cpu["wall_seconds"])
        )
        result["proxy_cpu_per_request_ms"] = per_req_ms
    result.setdefault("notes", []).append(note)

    final_path = out_dir / f"{label}.json"
    with open(final_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{label}] wrote {final_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "compare"], default="compare")
    p.add_argument(
        "--gil",
        choices=["on", "off"],
        default="on",
        help="(single mode only) which GIL state to run",
    )
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--proxy-port", type=int, default=4000)
    p.add_argument("--upstream-port", type=int, default=18080)
    p.add_argument("--out", default=str(HERE / "results"))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        python_bin = sys.executable
        if args.gil == "off":
            ft = _free_threaded_python()
            if ft is None:
                sys.exit(
                    "free-threaded python (e.g. python3.13t) not found on PATH; "
                    "install with `uv python install 3.13t`."
                )
            python_bin = ft
        run_one(
            label=f"gil_{args.gil}",
            python_bin=python_bin,
            gil=args.gil,
            proxy_port=args.proxy_port,
            upstream_port=args.upstream_port,
            duration=args.duration,
            concurrency=args.concurrency,
            out_dir=out_dir,
        )
        return

    # compare mode
    ft = _free_threaded_python()
    if ft is None:
        sys.exit(
            "free-threaded python (e.g. python3.13t) not found on PATH; "
            "install with `uv python install 3.13t`, then re-run."
        )

    on_result = run_one(
        label="gil_on",
        python_bin=sys.executable,
        gil="on",
        proxy_port=args.proxy_port,
        upstream_port=args.upstream_port,
        duration=args.duration,
        concurrency=args.concurrency,
        out_dir=out_dir,
    )
    # Stagger upstream port so a TIME_WAIT socket from the prior run
    # doesn't block bind.
    off_result = run_one(
        label="gil_off",
        python_bin=ft,
        gil="off",
        proxy_port=args.proxy_port,
        upstream_port=args.upstream_port + 1,
        duration=args.duration,
        concurrency=args.concurrency,
        out_dir=out_dir,
    )

    from tests.load_tests.free_threading_benchmark.report import format_report

    print()
    print(format_report(on_result, off_result))


if __name__ == "__main__":
    main()
