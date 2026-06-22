"""
Async load generator for the free-threading benchmark.

Drives a steady fan-out of streaming chat-completion requests at a litellm
proxy and records:

* TTFT  — time from request send to the first SSE byte received.
* RPS   — successful requests / wall-clock duration.
* Event-loop lag — sampled in-process from the load generator's own loop;
  this is a proxy signal for "is asyncio scheduling getting starved on
  this box". When the proxy and the load generator share a host (the
  default), proxy starvation also manifests as higher TTFT p99.

Output is a single JSON file consumable by `report.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import httpx


@dataclass
class Sample:
    ttft_ms: float
    total_ms: float
    status: int
    bytes_received: int


@dataclass
class Result:
    label: str
    duration_s: float
    target_concurrency: int
    sent: int
    completed: int
    errors: int
    rps: float
    ttft_ms_p50: float
    ttft_ms_p90: float
    ttft_ms_p99: float
    total_ms_p50: float
    total_ms_p99: float
    loop_lag_ms_p50: float
    loop_lag_ms_p99: float
    proxy_pid: Optional[int] = None
    proxy_cpu_seconds: Optional[float] = None
    proxy_cpu_per_request_ms: Optional[float] = None
    notes: List[str] = field(default_factory=list)


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def _event_loop_lag_probe(
    interval_s: float,
    samples: List[float],
    stop: asyncio.Event,
) -> None:
    """Records observed - expected sleep delta in ms."""
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
            return
        except asyncio.TimeoutError:
            pass
        elapsed = time.perf_counter() - t0
        samples.append(max(0.0, (elapsed - interval_s) * 1000.0))


async def _one_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
    out: List[Sample],
) -> None:
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    received = 0
    status = 0
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            status = resp.status_code
            async for chunk in resp.aiter_raw():
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000.0
                received += len(chunk)
    except Exception:
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000.0
    total = (time.perf_counter() - t0) * 1000.0
    out.append(
        Sample(
            ttft_ms=ttft or total,
            total_ms=total,
            status=status,
            bytes_received=received,
        )
    )


async def _worker(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
    deadline: float,
    samples: List[Sample],
) -> None:
    while time.perf_counter() < deadline:
        await _one_request(client, url, payload, headers, samples)


async def run_load(
    *,
    label: str,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    duration_s: float,
    warmup_s: float,
) -> Result:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": "ping"}],
    }

    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    limits = httpx.Limits(
        max_keepalive_connections=concurrency, max_connections=concurrency * 2
    )

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=False) as client:
        # Warmup — discarded.
        warm_samples: List[Sample] = []
        warm_deadline = time.perf_counter() + warmup_s
        warm_workers = [
            asyncio.create_task(
                _worker(client, url, payload, headers, warm_deadline, warm_samples)
            )
            for _ in range(min(concurrency, 8))
        ]
        await asyncio.gather(*warm_workers, return_exceptions=True)

        samples: List[Sample] = []
        loop_lag: List[float] = []
        stop = asyncio.Event()
        lag_task = asyncio.create_task(_event_loop_lag_probe(0.05, loop_lag, stop))

        deadline = time.perf_counter() + duration_s
        t_start = time.perf_counter()
        workers = [
            asyncio.create_task(
                _worker(client, url, payload, headers, deadline, samples)
            )
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers, return_exceptions=True)
        actual = time.perf_counter() - t_start
        stop.set()
        await lag_task

    completed = sum(1 for s in samples if 200 <= s.status < 300)
    errors = len(samples) - completed
    ttft = [s.ttft_ms for s in samples if 200 <= s.status < 300]
    total = [s.total_ms for s in samples if 200 <= s.status < 300]

    return Result(
        label=label,
        duration_s=actual,
        target_concurrency=concurrency,
        sent=len(samples),
        completed=completed,
        errors=errors,
        rps=completed / actual if actual > 0 else 0.0,
        ttft_ms_p50=_percentile(ttft, 50),
        ttft_ms_p90=_percentile(ttft, 90),
        ttft_ms_p99=_percentile(ttft, 99),
        total_ms_p50=_percentile(total, 50),
        total_ms_p99=_percentile(total, 99),
        loop_lag_ms_p50=_percentile(loop_lag, 50),
        loop_lag_ms_p99=_percentile(loop_lag, 99),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="e.g. gil_on / gil_off")
    p.add_argument("--base-url", default="http://127.0.0.1:4000")
    p.add_argument(
        "--api-key", default=os.environ.get("LITELLM_BENCH_API_KEY", "sk-bench")
    )
    p.add_argument("--model", default="mock-gpt")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--warmup", type=float, default=5.0)
    p.add_argument("--out", required=True, help="output json file")
    args = p.parse_args()

    result = asyncio.run(
        run_load(
            label=args.label,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            concurrency=args.concurrency,
            duration_s=args.duration,
            warmup_s=args.warmup,
        )
    )
    with open(args.out, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(
        f"wrote {args.out}: rps={result.rps:.1f} ttft_p50={result.ttft_ms_p50:.1f}ms "
        f"ttft_p99={result.ttft_ms_p99:.1f}ms loop_lag_p99={result.loop_lag_ms_p99:.1f}ms"
    )


if __name__ == "__main__":
    main()
