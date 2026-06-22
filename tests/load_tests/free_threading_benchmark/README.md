# Free-threading benchmark

This harness compares the LiteLLM proxy's throughput, latency, CPU usage,
and event-loop responsiveness on a stock CPython build vs. on a free-threaded
(no-GIL) CPython build (PEP 703 — see
[Python HOWTO: Free-Threaded Python](https://docs.python.org/3/howto/free-threading-python.html)).

It is **not** part of the regular test suite. It is opt-in, takes minutes
to run, and requires a free-threaded interpreter on `PATH` to compare both
states in one invocation.

## What it measures

For each interpreter mode the harness reports:

| metric                       | source                                                          |
| ---------------------------- | --------------------------------------------------------------- |
| `litellm total cpu`          | `/proc/<pid>/stat` `utime+stime` over a sampled wall window     |
| `per-request litellm cpu`    | total cpu / completed requests, normalized to the loadgen window |
| `ttft p50` / `p99`           | wall time from request send to first SSE byte received          |
| `rps`                        | successful streamed completions / wall-clock duration            |
| `event-loop lag p50` / `p99` | observed - expected sleep delta, sampled every 50 ms in the loadgen process |

The mock upstream (`mock_upstream.py`) holds the upstream-side TTFT and
inter-token gap **fixed**, so the difference between runs is litellm's own
work: request parsing, routing, callback dispatch, response serialization,
and the asyncio scheduling around it.

## What changes between runs

Only the interpreter and the `PYTHON_GIL` env var:

| label    | interpreter      | `PYTHON_GIL` |
| -------- | ---------------- | ------------ |
| `gil_on` | `sys.executable` | unset        |
| `gil_off`| `python3.13t`    | `0`          |

The orchestrator boots the proxy with `--num_workers 1` so the comparison
isolates intra-process concurrency. Multiple workers (forks) is the
GIL-on workaround that free-threading is meant to replace.

## Prerequisites

* Linux host (procfs CPU sampling).
* `uv` (or another way to install CPython 3.13t):
  ```
  uv python install 3.13t
  ln -s "$(uv python find 3.13t)" /usr/local/bin/python3.13t
  ```
* The proxy's runtime deps installed in your venv: `make install-proxy-dev`.
* Mock upstream uses `aiohttp`, which the proxy already depends on.

> **Heads up.** Several of litellm's transitive C-extension deps may not
> yet ship free-threaded wheels. When they're imported, CPython silently
> re-enables the GIL and prints a `RuntimeWarning`. The harness verifies
> `sys._is_gil_enabled()` at proxy startup and records the result in the
> `notes` field of each result file — always check it before trusting a
> comparison.

## Running

Compare both states (recommended):

```
make bench-freethreaded
# or, for finer control:
python -m tests.load_tests.free_threading_benchmark.run_benchmark \
    --mode compare --duration 180 --concurrency 64 \
    --out tests/load_tests/free_threading_benchmark/results
```

Single state (handy for iteration):

```
make bench-freethreaded-gil       # current interpreter, GIL on
make bench-freethreaded-nogil     # python3.13t, GIL off
```

Re-format an existing run:

```
python -m tests.load_tests.free_threading_benchmark.report \
    results/gil_on.json results/gil_off.json
```

## Interpreting the output

Example format (numbers fabricated for illustration):

```
==============================================================================
LiteLLM gateway — free-threading benchmark
==============================================================================
  duration:    180.0s vs 180.0s
  concurrency: 64
  completed:   gil_on=32400  gil_off=44820
  errors:      gil_on=0      gil_off=0

  metric                              gil_on        gil_off         Δ
  --------------------------------------------------------------------------
  litellm total cpu (sampled)           823.0s  →      370.0s    -55% (better)
  per-request litellm cpu                25.4ms →       14.1ms   -44% (better)
  ttft p50                             1600.0ms →      970.0ms   -39% (better)
  ttft p99                            11000.0ms →     2300.0ms   -79% (better)
  rps (sustained)                       415.0   →      573.0     +38% (better)
  event-loop lag p50                     12.0ms →        1.0ms   -92% (better)
  event-loop lag p99                    894.0ms →        8.0ms   -99% (better)
```

Things to watch for:

* **TTFT p99 collapsing while p50 barely moves** is the most reliable
  signal that the GIL was previously serializing CPU-bound work behind
  the event loop. Free-threading lets that work run on another core.
* **CPU goes *up* per request under free-threading** sometimes happens
  because reference counting now uses atomic operations. If `total cpu`
  drops while `per-request cpu` rises, throughput won.
* **Event-loop lag p99 dropping ~1-2 orders of magnitude** is the cleanest
  win: it means the scheduler stopped getting blocked behind C-extension
  work that was holding the GIL.

## Files

```
free_threading_benchmark/
  README.md            — this file
  proxy_config.yaml    — minimal proxy config pointing at the mock upstream
  mock_upstream.py     — OpenAI-compatible streaming mock
  loadgen.py           — async load generator + event-loop lag probe
  cpu_sampler.py       — procfs CPU sampler (process tree)
  report.py            — side-by-side formatter
  run_benchmark.py     — orchestrator
  results/             — output JSONs land here
```

## Caveats and known gaps

* Closed-loop load (fixed concurrency), not open-loop fixed RPS. RPS
  numbers should be read as "what concurrency=N sustains", not "what the
  proxy can handle if I aim for X qps".
* Single-host run: loadgen, proxy, and mock upstream share cores. For a
  cleaner number, pin each to its own cgroup or run the loadgen from a
  second host.
* The harness covers the gateway hot path (chat completions, streaming).
  It does not exercise `/v1/embeddings`, batch APIs, or guardrail hooks.
* No DB-backed run. Master-key-only auth keeps the comparison about the
  gateway, not Postgres. Re-run with `database_url` set if you want to
  capture the prisma path.
