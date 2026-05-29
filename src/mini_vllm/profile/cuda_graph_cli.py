from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional

import torch

try:
    from vllm.compilation.counter import compilation_counter
    from vllm.config import CUDAGraphMode
except ImportError as exc:
    if "undefined symbol" in str(exc) and "vllm" in str(exc):
        raise SystemExit(
            "Failed to import the vLLM CUDA extension.\n"
            f"Python executable: {sys.executable}\n"
            f"Runtime torch: {torch.__version__} "
            f"(CUDA {torch.version.cuda})\n"
            "The vLLM extension found on PYTHONPATH was built against a "
            "different torch/CUDA stack. Use the same environment that built "
            "3rdparty/vllm, or rebuild/install 3rdparty/vllm in this Python "
            "environment."
        ) from exc
    raise

from mini_vllm.model_runner import ModelRunner
from mini_vllm.profile.batch_sampler import BatchSampler, BatchSpec
from mini_vllm.profile.energy_meter import EnergyMeter, GpuFrequencyController
from mini_vllm.struct import Batch, Config
from mini_vllm.vllm_utils import get_vllm_config


PROFILED_PHASES = {"eager_no_cuda_graph", "replay_cuda_graph"}
EXPECTED_CUDA_GRAPH_ACTION = {
    "warmup_no_cuda_graph": "none",
    "eager_no_cuda_graph": "none",
    "capture_cuda_graph": "capture",
    "replay_cuda_graph": "replay",
}


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile mini_vllm CUDA graph warmup, capture, and replay phases."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--batch_config", type=str, required=True)
    parser.add_argument("--max_memory_utilization", type=float, default=0.8)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=None)

    parser.add_argument("--output", type=str, default="cuda_graph_profile.jsonl")
    parser.add_argument("--idle_s", type=float, default=0.0)
    parser.add_argument("--sample_interval_s", type=float, default=0.01)
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--no_sync_cuda", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--no_nvtx", action="store_true")
    parser.add_argument(
        "--no_cuda_profiler_capture",
        action="store_true",
        help=(
            "Do not wrap the eager and replay phases with cudaProfilerStart/Stop. "
            "Leave this unset when using nsys --capture-range=cudaProfilerApi."
        ),
    )
    parser.add_argument(
        "--nvtx_capture_name",
        type=str,
        default="cuda_graph_profile",
        help=(
            "Outer NVTX range name used for the eager and replay phases inside "
            "the captured Nsight regions."
        ),
    )

    parser.add_argument("--graphics_clock", type=str, default=None)
    parser.add_argument("--power_limit_w", type=int, default=None)

    return parser.parse_args(argv)


def _estimate_max_query_tokens(batch_config_path: str) -> Optional[int]:
    try:
        if batch_config_path.endswith(".json"):
            with open(batch_config_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            import yaml  # type: ignore

            with open(batch_config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(raw, list):
        return None
    max_tokens = 0
    for batch in raw:
        requests = batch.get("requests", [])
        total_query = 0
        for req in requests:
            try:
                total_query += int(req.get("query_len", 0))
            except Exception:
                pass
        if total_query > max_tokens:
            max_tokens = total_query
    return max_tokens if max_tokens > 0 else None


def _resolve_batch_config_path(batch_config_path: str) -> str:
    path = Path(batch_config_path)
    if path.exists():
        return str(path)

    source_root = Path(__file__).resolve().parents[2]
    candidates = [source_root / batch_config_path]

    parts = path.parts
    if parts[:2] == ("mini_vllm", "profile"):
        candidates.append(source_root / Path(*parts))
    if parts and parts[-1]:
        candidates.append(Path(__file__).resolve().parent / parts[-1])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return batch_config_path


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if not enabled or not torch.cuda.is_available():
        yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _sync_cuda(sync_cuda: bool) -> None:
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _counter_snapshot() -> dict[str, int]:
    return {
        "num_graphs_seen": compilation_counter.num_graphs_seen,
        "num_piecewise_capturable_graphs_seen": (
            compilation_counter.num_piecewise_capturable_graphs_seen
        ),
        "num_cudagraph_captured": compilation_counter.num_cudagraph_captured,
    }


def _batch_fields(
    batch: Batch,
    spec: Optional[BatchSpec],
    batch_index: int,
) -> dict[str, object]:
    total_context_tokens = sum(batch.context_lens)
    total_query_tokens = sum(batch.query_lens)
    fields: dict[str, object] = {
        "batch_index": batch_index,
        "num_reqs": len(batch.req_ids),
        "total_context_tokens": total_context_tokens,
        "total_query_tokens": total_query_tokens,
        "total_tokens": total_context_tokens + total_query_tokens,
        "total_query_len": total_query_tokens,
        "past_len": total_context_tokens,
        "sum_total_len_times_past_len": sum(
            (cl + ql) * cl for cl, ql in zip(batch.context_lens, batch.query_lens)
        ),
    }
    if spec is not None:
        fields.update(
            {
                "batch_name": spec.name,
                "batch_type": spec.batch_type,
            }
        )
    return fields


def _measure_phase(
    *,
    model_runner: ModelRunner,
    energy_meter: EnergyMeter,
    batch: Batch,
    phase: str,
    phase_index: int,
    cudagraph_runtime_mode: CUDAGraphMode | None,
    sync_cuda: bool,
    nvtx_enabled: bool,
    cuda_profiler_capture: bool,
    nvtx_capture_name: str,
) -> dict[str, object]:
    profiled_with_nvtx = phase in PROFILED_PHASES
    before = _counter_snapshot()

    _sync_cuda(sync_cuda)
    profiler_started = False
    if cuda_profiler_capture and profiled_with_nvtx and torch.cuda.is_available():
        torch.cuda.cudart().cudaProfilerStart()
        profiler_started = True

    energy_meter.start()
    t0 = time.perf_counter()
    samples = []
    try:
        with _nvtx_range(nvtx_capture_name, nvtx_enabled and profiled_with_nvtx):
            with _nvtx_range(phase, nvtx_enabled):
                model_runner.execute_batch(
                    batch,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                )
                _sync_cuda(sync_cuda)
        t1 = time.perf_counter()
    finally:
        if profiler_started:
            torch.cuda.cudart().cudaProfilerStop()
        samples = energy_meter.stop()

    after = _counter_snapshot()
    stats = energy_meter.summarize(samples)
    record: dict[str, object] = {
        "record_type": "cuda_graph_phase",
        "phase": phase,
        "phase_index": phase_index,
        "expected_cuda_graph_action": EXPECTED_CUDA_GRAPH_ACTION[phase],
        "requested_cudagraph_runtime_mode": (
            "auto" if cudagraph_runtime_mode is None else str(cudagraph_runtime_mode)
        ),
        "nvtx_profiled": profiled_with_nvtx,
        "cuda_profiler_captured": profiler_started,
        "latency_ms": (t1 - t0) * 1000.0,
        "avg_power_w": stats["avg_power_w"],
        "energy_j": stats["energy_j"],
        "energy_sample_duration_s": stats["duration_s"],
    }
    for key, value in before.items():
        record[f"{key}_before"] = value
    for key, value in after.items():
        record[f"{key}_after"] = value
        record[f"{key}_delta"] = value - before[key]
    if samples:
        record["graphics_clock_mhz"] = samples[-1].graphics_clock_mhz
        record["mem_clock_mhz"] = samples[-1].mem_clock_mhz
    return record


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator else None


def _safe_pct(delta: float, baseline: float) -> Optional[float]:
    return 100.0 * delta / baseline if baseline else None


def _summary_record(
    batch_fields: dict[str, object],
    phase_records: list[dict[str, object]],
) -> dict[str, object]:
    latencies = {
        str(row["phase"]): float(row["latency_ms"])
        for row in phase_records
    }
    warmup = latencies["warmup_no_cuda_graph"]
    eager = latencies["eager_no_cuda_graph"]
    capture = latencies["capture_cuda_graph"]
    replay = latencies["replay_cuda_graph"]

    latency_gain = eager - replay
    capture_overhead_vs_eager = capture - eager
    capture_overhead_vs_replay = capture - replay

    return {
        "record_type": "cuda_graph_summary",
        **batch_fields,
        "warmup_no_cuda_graph_latency_ms": warmup,
        "eager_no_cuda_graph_latency_ms": eager,
        "capture_cuda_graph_latency_ms": capture,
        "replay_cuda_graph_latency_ms": replay,
        "latency_gain_ms": latency_gain,
        "latency_gain_pct": _safe_pct(latency_gain, eager),
        "cuda_graph_speedup": _safe_ratio(eager, replay),
        "capture_overhead_vs_eager_ms": capture_overhead_vs_eager,
        "capture_overhead_vs_eager_pct": _safe_pct(capture_overhead_vs_eager, eager),
        "capture_overhead_vs_replay_ms": capture_overhead_vs_replay,
        "capture_to_replay_ratio": _safe_ratio(capture, replay),
    }


def _prime_dynamic_compile(
    *,
    model_runner: ModelRunner,
    batches: list[Batch],
    sync_cuda: bool,
) -> None:
    prime_batch = next((batch for batch in batches if sum(batch.query_lens) > 1), None)
    if prime_batch is None:
        return

    model_runner.execute_batch(
        prime_batch,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )
    _sync_cuda(sync_cuda)
    if hasattr(model_runner, "batch_descriptors"):
        model_runner.batch_descriptors.clear()


def _profile_batches(
    *,
    model_runner: ModelRunner,
    energy_meter: EnergyMeter,
    batches: Iterable[Batch],
    specs: Optional[Iterable[BatchSpec]],
    idle_s: float,
    sync_cuda: bool,
    show_progress: bool,
    nvtx_enabled: bool,
    cuda_profiler_capture: bool,
    nvtx_capture_name: str,
) -> list[dict[str, object]]:
    specs_list = list(specs) if specs is not None else []
    results: list[dict[str, object]] = []

    if idle_s > 0:
        samples = energy_meter.measure(idle_s)
        stats = energy_meter.summarize(samples)
        results.append(
            {
                "record_type": "idle",
                "idle_duration_s": idle_s,
                "idle_avg_power_w": stats["avg_power_w"],
                "idle_energy_j": stats["energy_j"],
            }
        )

    batches_iter = list(batches)
    _prime_dynamic_compile(
        model_runner=model_runner,
        batches=batches_iter,
        sync_cuda=sync_cuda,
    )

    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            batches_iter = tqdm(batches_iter, desc="Batches")
        except Exception:
            pass

    phase_plan: tuple[tuple[str, CUDAGraphMode | None], ...] = (
        ("warmup_no_cuda_graph", None),
        ("eager_no_cuda_graph", CUDAGraphMode.NONE),
        ("capture_cuda_graph", CUDAGraphMode.PIECEWISE),
        ("replay_cuda_graph", CUDAGraphMode.PIECEWISE),
    )

    for batch_index, batch in enumerate(batches_iter):
        spec = specs_list[batch_index] if batch_index < len(specs_list) else None
        fields = _batch_fields(batch, spec, batch_index)

        if hasattr(model_runner, "batch_descriptors"):
            model_runner.batch_descriptors.clear()

        phase_records: list[dict[str, object]] = []
        for phase_index, (phase, runtime_mode) in enumerate(phase_plan):
            record = _measure_phase(
                model_runner=model_runner,
                energy_meter=energy_meter,
                batch=batch,
                phase=phase,
                phase_index=phase_index,
                cudagraph_runtime_mode=runtime_mode,
                sync_cuda=sync_cuda,
                nvtx_enabled=nvtx_enabled,
                cuda_profiler_capture=cuda_profiler_capture,
                nvtx_capture_name=nvtx_capture_name,
            )
            record.update(fields)
            results.append(record)
            phase_records.append(record)

        results.append(_summary_record(fields, phase_records))

    return results


def _write_results(path: str, results: list[dict[str, object]]) -> None:
    if path.endswith(".csv"):
        keys = sorted({key for row in results for key in row.keys()})
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        return

    with open(path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    batch_config_path = _resolve_batch_config_path(args.batch_config)

    max_from_config = _estimate_max_query_tokens(batch_config_path)
    max_num_batched_tokens = (
        args.max_num_batched_tokens
        if args.max_num_batched_tokens is not None
        else max_from_config
    )

    config = Config(
        model_name=args.model_name,
        max_memory_utilization=args.max_memory_utilization,
        block_size=args.block_size,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    vllm_config = get_vllm_config(config)
    model_runner = ModelRunner(vllm_config)

    sampler = BatchSampler(block_size=model_runner.block_size, num_blocks=model_runner.num_blocks)
    specs = sampler.load_specs(batch_config_path)
    batches = sampler.build_batches(specs)

    energy_meter = EnergyMeter(
        device_index=args.device_index,
        sample_interval_s=args.sample_interval_s,
    )

    freq_ctrl = None
    try:
        if args.graphics_clock or args.power_limit_w is not None:
            freq_ctrl = GpuFrequencyController(device_index=args.device_index)
            if args.power_limit_w is not None:
                freq_ctrl.set_power_limit(args.power_limit_w)
            if args.graphics_clock:
                parts = args.graphics_clock.split(",")
                if len(parts) != 2:
                    raise ValueError("--graphics_clock expects MIN,MAX in MHz")
                freq_ctrl.set_graphics_clock(int(parts[0]), int(parts[1]))

        results = _profile_batches(
            model_runner=model_runner,
            energy_meter=energy_meter,
            batches=batches,
            specs=specs,
            idle_s=args.idle_s,
            sync_cuda=not args.no_sync_cuda,
            show_progress=not args.no_progress,
            nvtx_enabled=not args.no_nvtx,
            cuda_profiler_capture=not args.no_cuda_profiler_capture,
            nvtx_capture_name=args.nvtx_capture_name,
        )
        _write_results(args.output, results)
    finally:
        if freq_ctrl is not None:
            try:
                freq_ctrl.reset_graphics_clock()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
