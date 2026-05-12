"""
Side-by-side report formatter for the free-threading benchmark.

Takes two result dicts (gil-on, gil-off) and emits a fixed-width table
in the same shape as the example in the PR description.

Can also be run standalone:
    python -m tests.load_tests.free_threading_benchmark.report \
        results/gil_on.json results/gil_off.json
"""

from __future__ import annotations

import argparse
import json


def _delta_pct(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    pct = (after - before) / before * 100.0
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%"


def _row(
    label: str, before: float, after: float, unit: str, lower_is_better: bool
) -> str:
    delta = _delta_pct(before, after)
    arrow = ""
    if before > 0:
        improved = (after < before) if lower_is_better else (after > before)
        arrow = " (better)" if improved else " (worse)"
    return (
        f"  {label:<32} {before:>10.1f}{unit} → {after:>10.1f}{unit}   {delta}{arrow}"
    )


def format_report(on: dict, off: dict) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("LiteLLM gateway — free-threading benchmark")
    lines.append("=" * 78)
    lines.append(
        f"  duration:    {on.get('duration_s', 0):.1f}s vs {off.get('duration_s', 0):.1f}s"
    )
    lines.append(f"  concurrency: {on.get('target_concurrency')}")
    lines.append(
        f"  completed:   gil_on={on.get('completed')}  gil_off={off.get('completed')}"
    )
    lines.append(
        f"  errors:      gil_on={on.get('errors')}     gil_off={off.get('errors')}"
    )
    lines.append("")
    lines.append(
        "  metric                              gil_on        gil_off         Δ"
    )
    lines.append("  " + "-" * 74)

    cpu_on = on.get("proxy_cpu_seconds") or 0.0
    cpu_off = off.get("proxy_cpu_seconds") or 0.0
    lines.append(
        _row("litellm total cpu (sampled)", cpu_on, cpu_off, "s ", lower_is_better=True)
    )

    per_on = on.get("proxy_cpu_per_request_ms") or 0.0
    per_off = off.get("proxy_cpu_per_request_ms") or 0.0
    lines.append(
        _row("per-request litellm cpu", per_on, per_off, "ms", lower_is_better=True)
    )

    lines.append(
        _row(
            "ttft p50",
            on["ttft_ms_p50"],
            off["ttft_ms_p50"],
            "ms",
            lower_is_better=True,
        )
    )
    lines.append(
        _row(
            "ttft p99",
            on["ttft_ms_p99"],
            off["ttft_ms_p99"],
            "ms",
            lower_is_better=True,
        )
    )
    lines.append(
        _row("rps (sustained)", on["rps"], off["rps"], "  ", lower_is_better=False)
    )
    lines.append(
        _row(
            "event-loop lag p50",
            on["loop_lag_ms_p50"],
            off["loop_lag_ms_p50"],
            "ms",
            lower_is_better=True,
        )
    )
    lines.append(
        _row(
            "event-loop lag p99",
            on["loop_lag_ms_p99"],
            off["loop_lag_ms_p99"],
            "ms",
            lower_is_better=True,
        )
    )
    lines.append("")

    notes = (on.get("notes") or []) + (off.get("notes") or [])
    if notes:
        lines.append("notes:")
        for n in notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("on_json")
    p.add_argument("off_json")
    args = p.parse_args()
    with open(args.on_json) as f:
        on = json.load(f)
    with open(args.off_json) as f:
        off = json.load(f)
    print(format_report(on, off))


if __name__ == "__main__":
    main()
