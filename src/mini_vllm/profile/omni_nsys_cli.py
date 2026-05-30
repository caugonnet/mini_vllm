from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Optional

from mini_vllm.profile.cuda_graph_nsys_cli import (
    _avg,
    _max,
    _metrics_from_sqlite,
    _remove_stale_nsys_outputs,
    _supports_plural_gpu_metrics_device,
    _to_float,
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Nsight Systems over a vLLM-Omni multi-stage workload and export "
            "cross-stream concurrency plus sampled GPU utilization metrics. "
            "Unlike the single-stream cuda_graph profiler, omni runs CUDA work in "
            "separate worker processes, so this wrapper profiles the whole process "
            "tree (no cudaProfilerApi capture-range)."
        )
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--worker_backend", type=str, default="multi_process",
                        choices=["multi_process", "ray"])

    parser.add_argument("--output_prefix", type=str, default="omni_metrics")
    parser.add_argument("--phase_output", type=str, default=None)
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--summary_jsonl", type=str, default=None)
    parser.add_argument("--nsys_output", type=str, default=None)

    parser.add_argument("--nsys_path", type=str, default="nsys")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--cuda_visible_devices", type=str, default=None)
    parser.add_argument("--gpu_metrics_frequency", type=int, default=10000)
    parser.add_argument("--gpu_metrics_set", type=str, default=None)
    parser.add_argument("--cuda_graph_trace", choices=("graph", "node"), default="node")
    parser.add_argument("--idle_s", type=float, default=0.0)
    parser.add_argument("--graphics_clock", type=str, default=None)
    parser.add_argument("--power_limit_w", type=int, default=None)
    parser.add_argument("--skip_nsys_run", action="store_true")

    return parser.parse_args(argv)


def _run_nsys(args: argparse.Namespace) -> None:
    phase_output = Path(args.phase_output)
    nsys_output = Path(args.nsys_output)
    _remove_stale_nsys_outputs(nsys_output)
    if phase_output.exists():
        phase_output.unlink()

    gpu_metric_flag = (
        "--gpu-metrics-devices"
        if _supports_plural_gpu_metrics_device(args.nsys_path)
        else "--gpu-metrics-device"
    )

    cmd = [
        args.nsys_path,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--sample=none",
        "--cpuctxsw=none",
        f"--cuda-graph-trace={args.cuda_graph_trace}",
        f"{gpu_metric_flag}={args.device_index}",
        f"--gpu-metrics-frequency={args.gpu_metrics_frequency}",
        "--export=sqlite",
    ]
    if args.gpu_metrics_set:
        cmd.append(f"--gpu-metrics-set={args.gpu_metrics_set}")
    cmd.extend(
        [
            f"--output={nsys_output}",
            "--force-overwrite=true",
            args.python,
            "-m",
            "mini_vllm.profile.omni_profile_cli",
            "--model",
            args.model,
            "--workload",
            args.workload,
            "--output",
            str(phase_output),
            "--device_index",
            str(args.device_index),
            "--worker_backend",
            args.worker_backend,
            "--idle_s",
            str(args.idle_s),
        ]
    )
    if args.graphics_clock:
        cmd.extend(["--graphics_clock", args.graphics_clock])
    if args.power_limit_w is not None:
        cmd.extend(["--power_limit_w", str(args.power_limit_w)])

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[1]) for row in rows}


def _stream_concurrency_from_sqlite(path: Path) -> dict[str, object]:
    """Derive cross-stream concurrency from CUDA kernel intervals.

    Uses a sweep line over (start, end) kernel intervals to compute the time
    during which >= 2 kernels were active (genuine GPU concurrency), the union
    busy time, and the peak number of simultaneously active kernels. This is
    the signal that motivated the omni experiments: a single dense LLM forward
    has ~zero concurrent time, while a multi-stage omni pipeline overlaps work
    across stage streams.
    """
    if not path.exists():
        return {"concurrency_error": f"missing sqlite file: {path}"}

    conn = sqlite3.connect(path)
    try:
        table = "CUPTI_ACTIVITY_KIND_KERNEL"
        cols = _table_columns(conn, table)
        if not {"start", "end"}.issubset(cols):
            return {"concurrency_error": f"no kernel intervals in {table}"}

        stream_col = "streamId" if "streamId" in cols else None
        pid_col = "globalPid" if "globalPid" in cols else None

        select_cols = ["start", "end"]
        if stream_col:
            select_cols.append(stream_col)
        if pid_col:
            select_cols.append(pid_col)

        rows = conn.execute(
            f"select {', '.join(select_cols)} from {table} order by start"
        ).fetchall()
        if not rows:
            return {"concurrency_error": "no kernel rows"}

        events: list[tuple[int, int]] = []
        total_kernel_ns = 0
        streams: set = set()
        pids: set = set()
        for row in rows:
            start = int(row[0])
            end = int(row[1])
            if end < start:
                continue
            events.append((start, 1))
            events.append((end, -1))
            total_kernel_ns += end - start
            idx = 2
            if stream_col:
                streams.add(row[idx])
                idx += 1
            if pid_col:
                pids.add(row[idx])

        events.sort(key=lambda e: (e[0], -e[1]))
        active = 0
        max_active = 0
        prev_t: Optional[int] = None
        union_busy_ns = 0
        concurrent_ns = 0
        for t, delta in events:
            if prev_t is not None and active > 0:
                span = t - prev_t
                union_busy_ns += span
                if active >= 2:
                    concurrent_ns += span
            active += delta
            max_active = max(max_active, active)
            prev_t = t

        concurrency_ratio = (
            concurrent_ns / union_busy_ns if union_busy_ns > 0 else 0.0
        )
        overlap_factor = (
            total_kernel_ns / union_busy_ns if union_busy_ns > 0 else 0.0
        )
        return {
            "num_kernels": len(rows),
            "num_streams": len(streams) if stream_col else None,
            "num_pids": len(pids) if pid_col else None,
            "total_kernel_ms": total_kernel_ns / 1e6,
            "union_busy_ms": union_busy_ns / 1e6,
            "concurrent_ms": concurrent_ns / 1e6,
            "concurrency_ratio": concurrency_ratio,
            "overlap_factor": overlap_factor,
            "max_concurrent_kernels": max_active,
        }
    finally:
        conn.close()


def _gpu_metrics_new_schema(path: Path) -> Optional[dict[str, object]]:
    """Read sampled GPU utilization from the nsys >= 2024 ``GPU_METRICS`` table.

    Newer Nsight Systems stores GPU metrics as a ``(metricId, value)`` time
    series in ``GPU_METRICS`` with names in ``TARGET_INFO_GPU_METRICS`` instead
    of the legacy ``GENERIC_EVENTS`` JSON blobs. Returns ``None`` if this schema
    is not present so callers can fall back to the legacy reader.
    """
    if not path.exists():
        return None
    conn = sqlite3.connect(path)
    try:
        if not _table_columns(conn, "GPU_METRICS"):
            return None
        name_by_id = {
            int(mid): str(name)
            for mid, name in conn.execute(
                "select metricId, metricName from TARGET_INFO_GPU_METRICS"
            )
        }
        if not name_by_id:
            return None

        # Map our summary keys to the substring(s) of the metric names to sum.
        wanted = {
            "sm_active_pct": ("SMs Active [Throughput %]",),
            "warp_occupancy_pct": ("Compute Warps in Flight [Throughput %]",),
            "dram_bandwidth_pct": (
                "DRAM Read Bandwidth [Throughput %]",
                "DRAM Write Bandwidth [Throughput %]",
            ),
            "pcie_bandwidth_pct": (
                "PCIe RX Throughput [Throughput %]",
                "PCIe TX Throughput [Throughput %]",
            ),
        }
        ids_for = {
            key: [mid for mid, name in name_by_id.items() if name in names]
            for key, names in wanted.items()
        }

        per_metric: dict[int, list[float]] = {
            mid: [] for ids in ids_for.values() for mid in ids
        }
        if not per_metric:
            return None
        placeholders = ",".join("?" for _ in per_metric)
        for mid, value in conn.execute(
            f"select metricId, value from GPU_METRICS where metricId in ({placeholders})",
            tuple(per_metric),
        ):
            v = _to_float(value)
            if v is not None:
                per_metric[int(mid)].append(v)

        def _combined(ids: list[int], reduce) -> Optional[float]:
            # Sum the per-series reductions (e.g. read% + write%); skip empties.
            parts = [reduce(per_metric[mid]) for mid in ids if per_metric[mid]]
            parts = [p for p in parts if p is not None]
            return sum(parts) if parts else None

        sample_count = max((len(v) for v in per_metric.values()), default=0)
        return {
            "metric_schema": "gpu_metrics",
            "metric_sample_count": sample_count,
            "sm_active_pct_avg": _combined(ids_for["sm_active_pct"], _avg),
            "sm_active_pct_max": _combined(ids_for["sm_active_pct"], _max),
            "warp_occupancy_pct_avg": _combined(ids_for["warp_occupancy_pct"], _avg),
            "warp_occupancy_pct_max": _combined(ids_for["warp_occupancy_pct"], _max),
            "dram_bandwidth_pct_avg": _combined(ids_for["dram_bandwidth_pct"], _avg),
            "dram_bandwidth_pct_max": _combined(ids_for["dram_bandwidth_pct"], _max),
            "pcie_bandwidth_pct_avg": _combined(ids_for["pcie_bandwidth_pct"], _avg),
            "pcie_bandwidth_pct_max": _combined(ids_for["pcie_bandwidth_pct"], _max),
        }
    finally:
        conn.close()


def _gpu_metrics(path: Path) -> dict[str, object]:
    """GPU utilization summary, preferring the new schema with a legacy fallback."""
    new = _gpu_metrics_new_schema(path)
    if new is not None:
        return new
    return _metrics_from_sqlite(path)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _intersection_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if lo < hi:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _stage_nvtx_overlap_from_sqlite(path: Path) -> dict[str, object]:
    """Cross-stage concurrency derived from per-stage NVTX step ranges.

    Requires the run to have set ``VLLM_OMNI_NVTX_STAGES=1`` so each stage
    worker emits ``omni_stage{N}.step`` ranges. Computes, per stage, the busy
    time (union of its step ranges) and the pairwise time during which two or
    more stages were simultaneously stepping -- a direct, process-aware measure
    of how much the multi-stage pipeline overlaps work across stages.
    """
    if not path.exists():
        return {"stage_overlap_error": f"missing sqlite file: {path}"}

    conn = sqlite3.connect(path)
    try:
        cols = _table_columns(conn, "NVTX_EVENTS")
        if not {"start", "end", "text"}.issubset(cols):
            return {"stage_overlap_error": "no NVTX_EVENTS text ranges"}

        rows = conn.execute(
            "select text, start, end from NVTX_EVENTS "
            "where text like 'omni_stage%.step' and end is not null"
        ).fetchall()
        if not rows:
            return {
                "stage_overlap_note": (
                    "no omni_stage NVTX ranges (run with VLLM_OMNI_NVTX_STAGES=1)"
                )
            }

        by_stage: dict[str, list[tuple[int, int]]] = {}
        for text, start, end in rows:
            start = int(start)
            end = int(end)
            if end < start:
                continue
            by_stage.setdefault(str(text), []).append((start, end))

        merged = {stage: _merge_intervals(iv) for stage, iv in by_stage.items()}
        busy_ms = {
            stage: sum(e - s for s, e in iv) / 1e6 for stage, iv in merged.items()
        }
        step_counts = {stage: len(iv) for stage, iv in by_stage.items()}

        union_all = _merge_intervals(
            [iv for stage in merged.values() for iv in stage]
        )
        union_ms = sum(e - s for s, e in union_all) / 1e6

        # Pairwise concurrent time across distinct stages.
        stage_names = sorted(merged)
        pairwise: dict[str, float] = {}
        total_concurrent_ms = 0.0
        for idx, a in enumerate(stage_names):
            for b in stage_names[idx + 1 :]:
                inter_ms = _intersection_ns(merged[a], merged[b]) / 1e6
                pairwise[f"{a}|{b}"] = inter_ms
                total_concurrent_ms += inter_ms

        return {
            "num_stages_traced": len(merged),
            "stage_step_counts": step_counts,
            "stage_busy_ms": {k: round(v, 3) for k, v in busy_ms.items()},
            "stage_union_busy_ms": round(union_ms, 3),
            "stage_pairwise_concurrent_ms": {
                k: round(v, 3) for k, v in pairwise.items()
            },
            "stage_concurrent_ms": round(total_concurrent_ms, 3),
            "stage_concurrency_ratio": (
                round(total_concurrent_ms / union_ms, 4) if union_ms > 0 else 0.0
            ),
        }
    finally:
        conn.close()


def _sqlite_path(nsys_output: Path) -> Path:
    indexed = Path(f"{nsys_output}.sqlite")
    if indexed.exists():
        return indexed
    matches = sorted(nsys_output.parent.glob(f"{nsys_output.name}*.sqlite"))
    return matches[0] if matches else indexed


def _load_phase_summary(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {}
    if not path.exists():
        return summary
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record_type") == "omni_summary":
                summary = row
    return summary


def _build_summary_row(args: argparse.Namespace) -> dict[str, object]:
    phase_summary = _load_phase_summary(Path(args.phase_output))
    sqlite_path = _sqlite_path(Path(args.nsys_output))
    gpu_metrics = _gpu_metrics(sqlite_path)
    concurrency = _stream_concurrency_from_sqlite(sqlite_path)
    stage_overlap = _stage_nvtx_overlap_from_sqlite(sqlite_path)

    row: dict[str, object] = {
        "model": args.model,
        "workload": phase_summary.get("workload"),
        "num_prompts": phase_summary.get("num_prompts"),
        "num_outputs": phase_summary.get("num_outputs"),
        "total_wall_ms": phase_summary.get("total_wall_ms"),
        "avg_power_w": phase_summary.get("avg_power_w"),
        "energy_j": phase_summary.get("energy_j"),
        "sqlite": str(sqlite_path),
    }
    row.update(concurrency)
    row.update(stage_overlap)
    row.update(gpu_metrics)
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    output_prefix = Path(args.output_prefix)
    args.phase_output = args.phase_output or f"{output_prefix}.phases.jsonl"
    args.summary_csv = args.summary_csv or f"{output_prefix}.summary.csv"
    args.summary_jsonl = args.summary_jsonl or f"{output_prefix}.summary.jsonl"
    args.nsys_output = args.nsys_output or f"{output_prefix}.nsys"

    if not args.skip_nsys_run:
        _run_nsys(args)

    rows = [_build_summary_row(args)]
    _write_csv(Path(args.summary_csv), rows)
    _write_jsonl(Path(args.summary_jsonl), rows)

    print(f"Wrote {args.summary_csv}", file=sys.stderr)
    print(f"Wrote {args.summary_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
