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


EAGER_PHASE = "eager"
GRAPHED_PHASE = "graphed"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Nsight Systems CUDA graph profiling and export per-batch latency "
            "plus sampled GPU utilization metrics."
        )
    )

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_config", type=str, required=True)
    parser.add_argument("--max_memory_utilization", type=float, default=0.8)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--sample_interval_s", type=float, default=0.01)
    parser.add_argument("--idle_s", type=float, default=0.0)
    parser.add_argument("--graphics_clock", type=str, default=None)
    parser.add_argument("--power_limit_w", type=int, default=None)

    parser.add_argument("--output_prefix", type=str, default="cuda_graph_metrics")
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
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--skip_nsys_run", action="store_true")

    return parser.parse_args(argv)


def _load_batch_count(path: str) -> int:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    if not isinstance(raw, list):
        raise ValueError("batch config must be a list of batch specs")
    return len(raw)


def _supports_plural_gpu_metrics_device(nsys_path: str) -> bool:
    try:
        out = subprocess.run(
            [nsys_path, "profile", "--help"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout
    except Exception:
        return False
    return "--gpu-metrics-devices" in out


def _remove_stale_nsys_outputs(base: Path) -> None:
    parent = base.parent if str(base.parent) else Path(".")
    stem = base.name
    for suffix in ("*.nsys-rep", "*.sqlite", "*.qdstrm", "*.qdrep"):
        for path in parent.glob(f"{stem}.{suffix}"):
            path.unlink()
    for suffix in (".nsys-rep", ".sqlite", ".qdstrm", ".qdrep"):
        path = Path(f"{base}{suffix}")
        if path.exists():
            path.unlink()


def _run_nsys(args: argparse.Namespace, num_batches: int) -> None:
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
    capture_count = 2 * num_batches

    cmd = [
        args.nsys_path,
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        f"--cuda-graph-trace={args.cuda_graph_trace}",
        "--capture-range=cudaProfilerApi",
        f"--capture-range-end=repeat-shutdown:{capture_count}",
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
            "mini_vllm.profile.cuda_graph_cli",
            "--model_name",
            args.model_name,
            "--batch_config",
            args.batch_config,
            "--max_memory_utilization",
            str(args.max_memory_utilization),
            "--block_size",
            str(args.block_size),
            "--output",
            str(phase_output),
            "--idle_s",
            str(args.idle_s),
            "--sample_interval_s",
            str(args.sample_interval_s),
            "--device_index",
            str(args.device_index),
            "--no_progress",
        ]
    )
    if args.max_num_batched_tokens is not None:
        cmd.extend(["--max_num_batched_tokens", str(args.max_num_batched_tokens)])
    if args.graphics_clock:
        cmd.extend(["--graphics_clock", args.graphics_clock])
    if args.power_limit_w is not None:
        cmd.extend(["--power_limit_w", str(args.power_limit_w)])
    if args.no_progress:
        pass

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("Running:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, env=env)


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _avg(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _max(values: list[float]) -> Optional[float]:
    return max(values) if values else None


def _sum_fields(sample: dict[str, object], names: tuple[str, ...]) -> Optional[float]:
    values = [_to_float(sample.get(name)) for name in names]
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _gpu_capacity(conn: sqlite3.Connection) -> Optional[float]:
    try:
        row = conn.execute(
            "select smCount, maxWarpsPerSm from TARGET_INFO_GPU limit 1"
        ).fetchone()
    except sqlite3.Error:
        return None
    if not row:
        return None
    sm_count = _to_float(row[0])
    max_warps_per_sm = _to_float(row[1])
    if not sm_count or not max_warps_per_sm:
        return None
    return sm_count * max_warps_per_sm


def _metrics_from_sqlite(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "metric_sample_count": 0,
            "metric_error": f"missing sqlite file: {path}",
        }

    conn = sqlite3.connect(path)
    try:
        capacity = _gpu_capacity(conn)
        sm_active: list[float] = []
        warp_occupancy: list[float] = []
        dram_bandwidth: list[float] = []
        pcie_bandwidth: list[float] = []

        rows = conn.execute("select data from GENERIC_EVENTS")
        for (raw_data,) in rows:
            try:
                sample = json.loads(raw_data)
            except Exception:
                continue

            value = _to_float(sample.get("SM Active"))
            if value is not None:
                sm_active.append(value)

            value = _to_float(sample.get("Warp Occupancy"))
            if value is None:
                warps = _to_float(sample.get("Compute Warps In Flight"))
                if warps is not None and capacity:
                    value = 100.0 * warps / capacity
            if value is not None:
                warp_occupancy.append(value)

            value = _to_float(sample.get("DRAM Throughput"))
            if value is None:
                value = _sum_fields(
                    sample,
                    ("DRAM Read Throughput", "DRAM Write Throughput"),
                )
            if value is not None:
                dram_bandwidth.append(value)

            value = _to_float(sample.get("PCIe Throughput"))
            if value is None:
                value = _sum_fields(
                    sample,
                    ("PCIe RX Throughput", "PCIe TX Throughput"),
                )
            if value is not None:
                pcie_bandwidth.append(value)

        return {
            "metric_sample_count": len(sm_active)
            or len(warp_occupancy)
            or len(dram_bandwidth)
            or len(pcie_bandwidth),
            "sm_active_pct_avg": _avg(sm_active),
            "sm_active_pct_max": _max(sm_active),
            "warp_occupancy_pct_avg": _avg(warp_occupancy),
            "warp_occupancy_pct_max": _max(warp_occupancy),
            "dram_bandwidth_pct_avg": _avg(dram_bandwidth),
            "dram_bandwidth_pct_max": _max(dram_bandwidth),
            "pcie_bandwidth_pct_avg": _avg(pcie_bandwidth),
            "pcie_bandwidth_pct_max": _max(pcie_bandwidth),
        }
    finally:
        conn.close()


def _load_phase_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sqlite_path(nsys_output: Path, capture_index: int) -> Path:
    indexed = Path(f"{nsys_output}.{capture_index}.sqlite")
    if indexed.exists():
        return indexed
    return Path(f"{nsys_output}.sqlite")


def _prefixed(prefix: str, row: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def _build_summary_rows(
    *,
    phase_output: Path,
    nsys_output: Path,
    num_batches: int,
) -> list[dict[str, object]]:
    phase_rows = _load_phase_rows(phase_output)
    summaries = {
        int(row["batch_index"]): row
        for row in phase_rows
        if row.get("record_type") == "cuda_graph_summary"
    }

    rows: list[dict[str, object]] = []
    for batch_index in range(num_batches):
        summary = summaries[batch_index]
        eager_sqlite = _sqlite_path(nsys_output, 2 * batch_index + 1)
        graphed_sqlite = _sqlite_path(nsys_output, 2 * batch_index + 2)
        eager_metrics = _metrics_from_sqlite(eager_sqlite)
        graphed_metrics = _metrics_from_sqlite(graphed_sqlite)

        row = {
            "batch_index": summary.get("batch_index"),
            "batch_name": summary.get("batch_name"),
            "batch_type": summary.get("batch_type"),
            "num_reqs": summary.get("num_reqs"),
            "total_context_tokens": summary.get("total_context_tokens"),
            "total_query_tokens": summary.get("total_query_tokens"),
            "total_tokens": summary.get("total_tokens"),
            "eager_time_ms": summary.get("eager_no_cuda_graph_latency_ms"),
            "capture_time_ms": summary.get("capture_cuda_graph_latency_ms"),
            "graphed_time_ms": summary.get("replay_cuda_graph_latency_ms"),
            "latency_gain_ms": summary.get("latency_gain_ms"),
            "latency_gain_pct": summary.get("latency_gain_pct"),
            "cuda_graph_speedup": summary.get("cuda_graph_speedup"),
            "capture_overhead_vs_eager_ms": summary.get("capture_overhead_vs_eager_ms"),
            "capture_overhead_vs_replay_ms": summary.get("capture_overhead_vs_replay_ms"),
            "eager_sqlite": str(eager_sqlite),
            "graphed_sqlite": str(graphed_sqlite),
        }
        row.update(_prefixed(EAGER_PHASE, eager_metrics))
        row.update(_prefixed(GRAPHED_PHASE, graphed_metrics))

        # Unprefixed aliases use the graphed replay metrics, which are usually the
        # values used for CUDA graph steady-state comparisons.
        row.update(
            {
                "sm_active_pct_avg": graphed_metrics.get("sm_active_pct_avg"),
                "warp_occupancy_pct_avg": graphed_metrics.get(
                    "warp_occupancy_pct_avg"
                ),
                "dram_bandwidth_pct_avg": graphed_metrics.get(
                    "dram_bandwidth_pct_avg"
                ),
                "pcie_bandwidth_pct_avg": graphed_metrics.get(
                    "pcie_bandwidth_pct_avg"
                ),
            }
        )
        rows.append(row)
    return rows


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

    num_batches = _load_batch_count(args.batch_config)
    if num_batches <= 0:
        raise ValueError("batch config contains no batches")

    if not args.skip_nsys_run:
        _run_nsys(args, num_batches)

    rows = _build_summary_rows(
        phase_output=Path(args.phase_output),
        nsys_output=Path(args.nsys_output),
        num_batches=num_batches,
    )
    _write_csv(Path(args.summary_csv), rows)
    _write_jsonl(Path(args.summary_jsonl), rows)

    print(f"Wrote {args.summary_csv}", file=sys.stderr)
    print(f"Wrote {args.summary_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
