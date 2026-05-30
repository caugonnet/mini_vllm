from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional

from mini_vllm.profile import omni_workload
from mini_vllm.profile.energy_meter import EnergyMeter, GpuFrequencyController


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive a vLLM-Omni multi-stage pipeline as a profiling workload. "
            "Measures end-to-end and per-stage-output latency plus power/energy, "
            "and optionally enables vLLM-Omni's per-worker torch profiler."
        )
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--output", type=str, default="omni_profile.jsonl")

    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--sample_interval_s", type=float, default=0.01)
    parser.add_argument("--idle_s", type=float, default=0.0)

    parser.add_argument("--worker_backend", type=str, default="multi_process",
                        choices=["multi_process", "ray"])
    parser.add_argument("--stage_init_timeout", type=int, default=300)
    parser.add_argument("--init_timeout", type=int, default=600)
    parser.add_argument("--log_stats", action="store_true")
    parser.add_argument(
        "--omni_arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra keyword forwarded to Omni(...) (repeatable). Values are parsed as JSON when possible.",
    )

    parser.add_argument(
        "--torch_profiler_dir",
        type=str,
        default=None,
        help="If set, exports VLLM_TORCH_PROFILER_DIR and runs Omni.start_profile/stop_profile.",
    )
    parser.add_argument(
        "--profile_stages",
        type=str,
        default=None,
        help="Comma-separated stage ids to torch-profile (default: all stages).",
    )
    parser.add_argument(
        "--trace_flush_s",
        type=float,
        default=30.0,
        help="Seconds to wait after stop_profile for workers to flush traces.",
    )

    parser.add_argument("--no_nvtx", action="store_true")
    parser.add_argument(
        "--cuda_profiler_capture",
        action="store_true",
        help=(
            "Wrap generation with cudaProfilerStart/Stop in the orchestrator. "
            "Note: omni CUDA work runs in worker processes, so this only gates "
            "the orchestrator context; the omni nsys wrapper does not rely on it."
        ),
    )

    parser.add_argument("--graphics_clock", type=str, default=None)
    parser.add_argument("--power_limit_w", type=int, default=None)

    return parser.parse_args(argv)


def _parse_omni_args(pairs: list[str]) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--omni_arg expects KEY=VALUE, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            value: object = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        kwargs[key.strip()] = value
    return kwargs


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if not enabled:
        yield
        return
    try:
        import torch

        if not torch.cuda.is_available():
            yield
            return
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    except Exception:
        yield


@contextmanager
def _cuda_profiler(enabled: bool):
    started = False
    try:
        if enabled:
            import torch

            if torch.cuda.is_available():
                torch.cuda.cudart().cudaProfilerStart()
                started = True
        yield
    finally:
        if started:
            import torch

            torch.cuda.cudart().cudaProfilerStop()


def _import_omni():
    try:
        from vllm_omni.entrypoints.omni import Omni  # type: ignore

        return Omni
    except ImportError as exc:
        raise SystemExit(
            "Failed to import vllm_omni.entrypoints.omni.Omni.\n"
            f"Python executable: {sys.executable}\n"
            "Install vLLM-Omni (see README 'Optional: vLLM-Omni') and make sure "
            "it is on PYTHONPATH alongside the matching vllm build."
        ) from exc


def _build_sampling_params(omni, spec: omni_workload.OmniWorkloadSpec):
    """Use Omni's per-stage defaults, capping the thinker (stage 0) tokens."""
    defaults = getattr(omni, "default_sampling_params_list", None)
    if not defaults:
        return None
    params = [copy.deepcopy(p) for p in defaults]
    head = params[0]
    if spec.max_tokens and hasattr(head, "max_tokens"):
        head.max_tokens = spec.max_tokens
    if hasattr(head, "temperature"):
        head.temperature = spec.temperature
    return params


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    spec = omni_workload.load_spec(args.workload)
    prompts = omni_workload.build_prompts(spec)

    torch_profile = bool(args.torch_profiler_dir)
    if torch_profile:
        os.makedirs(args.torch_profiler_dir, exist_ok=True)
        os.environ["VLLM_TORCH_PROFILER_DIR"] = args.torch_profiler_dir
    profile_stages = (
        [int(s) for s in args.profile_stages.split(",") if s.strip()]
        if args.profile_stages
        else None
    )

    Omni = _import_omni()

    omni_kwargs: dict[str, object] = {
        "model": args.model,
        "worker_backend": args.worker_backend,
        "stage_init_timeout": args.stage_init_timeout,
        "init_timeout": args.init_timeout,
        "log_stats": args.log_stats,
    }
    omni_kwargs.update(_parse_omni_args(args.omni_arg))

    energy_meter = EnergyMeter(
        device_index=args.device_index,
        sample_interval_s=args.sample_interval_s,
    )

    freq_ctrl = None
    results: list[dict[str, object]] = []
    omni = None
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

        if args.idle_s > 0:
            idle_samples = energy_meter.measure(args.idle_s)
            idle_stats = energy_meter.summarize(idle_samples)
            results.append(
                {
                    "record_type": "idle",
                    "idle_duration_s": args.idle_s,
                    "idle_avg_power_w": idle_stats["avg_power_w"],
                    "idle_energy_j": idle_stats["energy_j"],
                }
            )

        omni = Omni(**omni_kwargs)
        num_stages = getattr(omni, "num_stages", None)
        sampling_params_list = _build_sampling_params(omni, spec)

        results.append(
            {
                "record_type": "omni_config",
                "model": args.model,
                "workload": spec.name,
                "query_type": spec.query_type,
                "modalities": spec.modalities,
                "num_prompts": len(prompts),
                "num_stages": num_stages,
                "worker_backend": args.worker_backend,
                "max_tokens": spec.max_tokens,
            }
        )

        if torch_profile and hasattr(omni, "start_profile"):
            omni.start_profile(stages=profile_stages)

        energy_meter.start()
        t0 = time.perf_counter()
        with _cuda_profiler(args.cuda_profiler_capture):
            with _nvtx_range("omni_generate", not args.no_nvtx):
                generator = omni.generate(
                    prompts, sampling_params_list, py_generator=True
                )
                for index, stage_outputs in enumerate(generator):
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    output_type = getattr(stage_outputs, "final_output_type", None)
                    request_output = getattr(stage_outputs, "request_output", None)
                    request_id = getattr(request_output, "request_id", None)
                    results.append(
                        {
                            "record_type": "omni_stage_output",
                            "output_index": index,
                            "request_id": request_id,
                            "output_type": output_type,
                            "elapsed_ms": elapsed_ms,
                        }
                    )
        t1 = time.perf_counter()
        samples = energy_meter.stop()

        if torch_profile and hasattr(omni, "stop_profile"):
            omni.stop_profile(stages=profile_stages)
            if args.trace_flush_s > 0:
                time.sleep(args.trace_flush_s)

        stats = energy_meter.summarize(samples)
        num_outputs = sum(
            1 for r in results if r.get("record_type") == "omni_stage_output"
        )
        results.append(
            {
                "record_type": "omni_summary",
                "model": args.model,
                "workload": spec.name,
                "num_prompts": len(prompts),
                "num_outputs": num_outputs,
                "total_wall_ms": (t1 - t0) * 1000.0,
                "avg_power_w": stats["avg_power_w"],
                "energy_j": stats["energy_j"],
                "energy_sample_duration_s": stats["duration_s"],
                "torch_profiler_dir": args.torch_profiler_dir,
            }
        )
    finally:
        if omni is not None:
            try:
                omni.close()
            except Exception:
                pass
        if freq_ctrl is not None:
            try:
                freq_ctrl.reset_graphics_clock()
            except Exception:
                pass

    _write_results(args.output, results)
    return 0


def _write_results(path: str, results: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
