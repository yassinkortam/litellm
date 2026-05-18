#!/usr/bin/env python3
"""
Free-threading vs uvicorn-workers benchmark for the LiteLLM proxy.

Compares two ways of running the proxy with the same parallelism budget:

  * workers         : `litellm --num_workers N`                  (N processes)
  * free-threading  : `litellm --num_workers N --run_free_threading`
                      (N event-loop threads in one process; true CPU
                      parallelism on a free-threaded Python 3.13+ build,
                      see https://docs.python.org/3/howto/free-threading-python.html)

For each mode it drives load against `/v1/chat/completions` and
`/v1/messages` and reports:

  * non-streaming : RPS and latency (mean / p50 / p90 / p99)
  * streaming     : TTFT (time to first token) and TPM (tokens per minute)

Both models use `mock_response` (see benchmark_config.yaml) so there is no upstream
provider call: the only thing that varies between modes is the proxy's
serving layer.

Examples
--------
Benchmark both modes with 4-way parallelism (the headline comparison)::

    python benchmark.py --mode both --num-workers 4

Benchmark a proxy you started yourself::

    litellm --config benchmark_config.yaml --num_workers 4 --run_free_threading --port 4000
    python benchmark.py --no-launch --mode free-threading --port 4000

Run `python benchmark.py --help` for all options.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import httpx
except ImportError:  # pragma: no cover - httpx is a hard litellm dependency
    print(
        "httpx is required to run this benchmark. Install it with "
        "`pip install httpx` (it ships with `litellm[proxy]`).",
        file=sys.stderr,
    )
    raise SystemExit(1)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "benchmark_config.yaml")
MASTER_KEY = "sk-benchmark-master-key"

CHAT_PROMPT = "Summarize the benefits of free-threading in two sentences."
MSG_PROMPT = "Summarize the benefits of free-threading in two sentences."


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass
class PhaseResult:
    label: str
    streaming: bool
    duration_s: float
    completed: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)  # full request time
    ttft_ms: list[float] = field(default_factory=list)  # streaming only
    total_tokens: int = 0

    @property
    def rps(self) -> float:
        return self.completed / self.duration_s if self.duration_s else 0.0

    @property
    def tpm(self) -> float:
        return (self.total_tokens / self.duration_s * 60.0) if self.duration_s else 0.0

    def summary(self) -> str:
        if not self.streaming:
            lat = self.latencies_ms
            return (
                f"RPS={self.rps:8.1f}  "
                f"lat ms mean={statistics.mean(lat) if lat else 0:7.1f} "
                f"p50={_percentile(lat, 0.50):7.1f} "
                f"p90={_percentile(lat, 0.90):7.1f} "
                f"p99={_percentile(lat, 0.99):7.1f}  "
                f"ok={self.completed} err={self.errors}"
            )
        ttft = self.ttft_ms
        return (
            f"TPM={self.tpm:10.0f}  "
            f"TTFT ms mean={statistics.mean(ttft) if ttft else 0:7.1f} "
            f"p50={_percentile(ttft, 0.50):7.1f} "
            f"p90={_percentile(ttft, 0.90):7.1f} "
            f"p99={_percentile(ttft, 0.99):7.1f}  "
            f"ok={self.completed} err={self.errors}"
        )


# --------------------------------------------------------------------------- #
# Load generation
# --------------------------------------------------------------------------- #
def _count_tokens(text: str) -> int:
    # Whitespace-token approximation. Good enough for a *relative* TPM
    # comparison since the mock body is fixed across modes.
    return len(text.split())


async def _one_nonstream(
    client: httpx.AsyncClient, url: str, payload: dict, headers: dict, res: PhaseResult
) -> None:
    start = time.perf_counter()
    try:
        r = await client.post(url, json=payload, headers=headers)
        if r.status_code != 200:
            res.errors += 1
            return
        res.latencies_ms.append((time.perf_counter() - start) * 1000.0)
        res.completed += 1
    except Exception:
        res.errors += 1


async def _one_stream(
    client: httpx.AsyncClient, url: str, payload: dict, headers: dict, res: PhaseResult
) -> None:
    start = time.perf_counter()
    first_token_at: Optional[float] = None
    text_parts: list[str] = []
    usage_tokens = 0
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as r:
            if r.status_code != 200:
                await r.aread()
                res.errors += 1
                return
            ctype = r.headers.get("content-type", "")
            if "text/event-stream" in ctype:
                async for line in r.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    chunk = _safe_json(data)
                    if chunk is None:
                        continue
                    text_parts.append(_extract_stream_text(chunk))
                    usage_tokens = max(usage_tokens, _extract_usage_tokens(chunk))
            else:
                # The Anthropic mock path returns a single non-streamed JSON
                # body even when stream=True. Treat the whole body as one
                # "first token" so the phase still produces TTFT/TPM numbers.
                body = await r.aread()
                first_token_at = time.perf_counter()
                chunk = _safe_json(body.decode() or "{}") or {}
                text_parts.append(_extract_stream_text(chunk))
                usage_tokens = _extract_usage_tokens(chunk)
    except Exception:
        res.errors += 1
        return

    if first_token_at is None:
        res.errors += 1
        return
    res.ttft_ms.append((first_token_at - start) * 1000.0)
    res.latencies_ms.append((time.perf_counter() - start) * 1000.0)
    res.total_tokens += usage_tokens or _count_tokens("".join(text_parts))
    res.completed += 1


def _safe_json(data: str) -> Optional[dict]:
    import json

    try:
        return json.loads(data)
    except Exception:
        return None


def _extract_stream_text(chunk: dict) -> str:
    # OpenAI chat streaming
    choices = chunk.get("choices")
    if choices:
        delta = choices[0].get("delta") or {}
        if delta.get("content"):
            return str(delta["content"])
        msg = choices[0].get("message") or {}
        if msg.get("content"):
            return str(msg["content"])
    # Anthropic messages (streamed delta or full mock body)
    if chunk.get("type") == "content_block_delta":
        return str((chunk.get("delta") or {}).get("text", ""))
    content = chunk.get("content")
    if isinstance(content, list):
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return ""


def _extract_usage_tokens(chunk: dict) -> int:
    usage = chunk.get("usage") or {}
    if "completion_tokens" in usage:
        return int(usage["completion_tokens"])
    if "output_tokens" in usage:
        return int(usage["output_tokens"])
    return 0


async def _run_phase(
    base_url: str,
    label: str,
    path: str,
    payload: dict,
    streaming: bool,
    duration_s: float,
    concurrency: int,
) -> PhaseResult:
    res = PhaseResult(label=label, streaming=streaming, duration_s=duration_s)
    headers = {
        "Authorization": f"Bearer {MASTER_KEY}",
        "anthropic-version": "2023-06-01",
    }
    url = base_url + path
    deadline = time.perf_counter() + duration_s
    limits = httpx.Limits(
        max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8
    )
    timeout = httpx.Timeout(60.0)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:

        async def worker() -> None:
            while time.perf_counter() < deadline:
                if streaming:
                    await _one_stream(client, url, payload, headers, res)
                else:
                    await _one_nonstream(client, url, payload, headers, res)

        actual_start = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        res.duration_s = time.perf_counter() - actual_start
    return res


# --------------------------------------------------------------------------- #
# Proxy lifecycle
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def launch_proxy(
    mode: str,
    *,
    litellm_bin: str,
    config: str,
    host: str,
    port: int,
    num_workers: int,
):
    cmd = [
        litellm_bin,
        "--config",
        config,
        "--host",
        host,
        "--port",
        str(port),
        "--num_workers",
        str(num_workers),
    ]
    if mode == "free-threading":
        cmd.append("--run_free_threading")

    env = {**os.environ, "LITELLM_MODE": "PRODUCTION"}
    print(f"\n>>> launching proxy [{mode}]: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        _wait_ready(host, port, proc)
        yield
    finally:
        print(f">>> stopping proxy [{mode}] (pid {proc.pid})")
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)


def _wait_ready(host: str, port: int, proc: subprocess.Popen, timeout_s: int = 90):
    url = f"http://{host}:{port}/health/liveliness"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"proxy exited during startup with code {proc.returncode}"
            )
        with contextlib.suppress(Exception):
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                print(f">>> proxy ready at {host}:{port}")
                time.sleep(1.0)  # let all workers/threads settle
                return
        time.sleep(0.5)
    raise RuntimeError(f"proxy not ready within {timeout_s}s")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
PHASES = [
    ("chat  /v1/chat/completions  non-stream", "/v1/chat/completions", False, "chat"),
    ("chat  /v1/chat/completions  stream    ", "/v1/chat/completions", True, "chat"),
    ("msgs  /v1/messages          non-stream", "/v1/messages", False, "messages"),
    ("msgs  /v1/messages          stream    ", "/v1/messages", True, "messages"),
]


def _payload(kind: str, streaming: bool) -> dict:
    if kind == "chat":
        body: dict[str, Any] = {
            "model": "bench-chat",
            "messages": [{"role": "user", "content": CHAT_PROMPT}],
            "max_tokens": 128,
        }
        if streaming:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body
    body = {
        "model": "bench-messages",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": MSG_PROMPT}],
    }
    if streaming:
        body["stream"] = True
    return body


async def benchmark_mode(
    mode: str, base_url: str, duration: float, concurrency: int, warmup: float
) -> list[PhaseResult]:
    # Warmup so JIT/imports/connection pools don't skew the first numbers.
    if warmup > 0:
        print(f">>> warmup {warmup:.0f}s ...")
        await _run_phase(
            base_url,
            "warmup",
            "/v1/chat/completions",
            _payload("chat", False),
            False,
            warmup,
            concurrency,
        )

    results: list[PhaseResult] = []
    for label, path, streaming, kind in PHASES:
        print(f">>> [{mode}] {label} ({duration:.0f}s @ concurrency {concurrency})")
        res = await _run_phase(
            base_url,
            label,
            path,
            _payload(kind, streaming),
            streaming,
            duration,
            concurrency,
        )
        print(f"    {res.summary()}")
        results.append(res)
    return results


def print_comparison(all_results: dict[str, list[PhaseResult]]) -> None:
    print("\n" + "=" * 100)
    print("RESULTS")
    print("=" * 100)
    for mode, results in all_results.items():
        print(f"\n[{mode}]")
        for res in results:
            print(f"  {res.label}  {res.summary()}")

    if "workers" in all_results and "free-threading" in all_results:
        print("\n" + "-" * 100)
        print("free-threading vs workers (positive = free-threading better)")
        print("-" * 100)
        w = {r.label: r for r in all_results["workers"]}
        f = {r.label: r for r in all_results["free-threading"]}
        for label in w:
            rw, rf = w[label], f[label]
            if rw.streaming:
                base = rw.tpm or 1.0
                delta = (rf.tpm - rw.tpm) / base * 100.0
                print(
                    f"  {label}  TPM {rw.tpm:9.0f} -> {rf.tpm:9.0f}  "
                    f"({delta:+.1f}%)"
                )
            else:
                base = rw.rps or 1.0
                delta = (rf.rps - rw.rps) / base * 100.0
                print(
                    f"  {label}  RPS {rw.rps:8.1f} -> {rf.rps:8.1f}  "
                    f"({delta:+.1f}%)"
                )
    print("=" * 100)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark LiteLLM proxy: uvicorn workers vs free-threading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["workers", "free-threading", "both"],
        default="both",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=4000)
    p.add_argument("--duration", type=float, default=20.0, help="seconds per phase")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--warmup", type=float, default=5.0, help="warmup seconds")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--litellm-bin", default="litellm")
    p.add_argument(
        "--no-launch",
        action="store_true",
        help="benchmark an already-running proxy instead of launching one",
    )
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    base_url = f"http://{args.host}:{args.port}"
    modes = ["workers", "free-threading"] if args.mode == "both" else [args.mode]
    all_results: dict[str, list[PhaseResult]] = {}

    for mode in modes:
        if args.no_launch:
            all_results[mode] = await benchmark_mode(
                mode, base_url, args.duration, args.concurrency, args.warmup
            )
        else:
            with launch_proxy(
                mode,
                litellm_bin=args.litellm_bin,
                config=args.config,
                host=args.host,
                port=args.port,
                num_workers=args.num_workers,
            ):
                all_results[mode] = await benchmark_mode(
                    mode, base_url, args.duration, args.concurrency, args.warmup
                )

    print_comparison(all_results)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
