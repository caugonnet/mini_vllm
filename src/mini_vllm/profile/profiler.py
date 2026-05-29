from __future__ import annotations

import csv
import json
import time
from typing import Iterable, Optional

import torch
from vllm.compilation.counter import compilation_counter

from mini_vllm.profile.batch_sampler import BatchSpec
from mini_vllm.profile.energy_meter import EnergyMeter
from mini_vllm.struct import Batch


class BatchProfiler:
    def __init__(
        self,
        model_runner,
        energy_meter: EnergyMeter,
        warmup: int = 2,
        repeats: int = 10,
        sync_cuda: bool = True,
        show_progress: bool = True,
    ):
        self.model_runner = model_runner
        self.energy_meter = energy_meter
        self.warmup = warmup
        self.repeats = repeats
        self.sync_cuda = sync_cuda
        self.show_progress = show_progress

    def profile(
        self,
        batches: Iterable[Batch],
        specs: Optional[Iterable[BatchSpec]] = None,
        output_path: Optional[str] = None,
        idle_s: float = 2.0,
    ) -> list[dict[str, object]]:
        specs_list = list(specs) if specs is not None else []
        results: list[dict[str, object]] = []

        idle_stats = self._measure_idle(idle_s) if idle_s > 0 else None
        if idle_stats is not None:
            results.append({"record_type": "idle", **idle_stats})

        batches_iter = list(batches)
        outer = self._wrap_progress(batches_iter, desc="Batches")
        for idx, batch in enumerate(outer):
            spec = specs_list[idx] if idx < len(specs_list) else None
            self._warmup(batch)
            reps = range(self.repeats)
            inner = self._wrap_progress(reps, desc=f"Repeats batch {idx + 1}")
            for rep in inner:
                record = self._measure_batch(batch)
                total_context_tokens = sum(batch.context_lens)
                total_query_tokens = sum(batch.query_lens)
                sum_total_len_times_past_len = sum(
                    (cl + ql) * cl for cl, ql in zip(batch.context_lens, batch.query_lens)
                )
                record.update(
                    {
                        "record_type": "batch",
                        "repeat": rep,
                        "batch_index": idx,
                        "num_reqs": len(batch.req_ids),
                        "total_context_tokens": total_context_tokens,
                        "total_query_tokens": total_query_tokens,
                        "total_tokens": total_context_tokens + total_query_tokens,
                        "total_query_len": total_query_tokens,
                        "past_len": total_context_tokens,
                        "sum_total_len_times_past_len": sum_total_len_times_past_len,
                    }
                )
                if spec is not None:
                    record.update(
                        {
                            "batch_name": spec.name,
                            "batch_type": spec.batch_type,
                        }
                    )
                results.append(record)

        if output_path:
            self._write_results(output_path, results)
        return results

    def _warmup(self, batch: Batch) -> None:
        for _ in range(self.warmup):
            self.model_runner.execute_batch(batch)
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _measure_batch(self, batch: Batch) -> dict[str, object]:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.energy_meter.start()
        t0 = time.perf_counter()
        _ = self.model_runner.execute_batch(batch)
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples = self.energy_meter.stop()
        stats = self.energy_meter.summarize(samples)

        latency_ms = (t1 - t0) * 1000.0
        record = {
            "latency_ms": latency_ms,
            "avg_power_w": stats["avg_power_w"],
            "energy_j": stats["energy_j"],
            "num_graphs_seen": compilation_counter.num_graphs_seen,
            "num_piecewise_capturable_graphs_seen": (
                compilation_counter.num_piecewise_capturable_graphs_seen
            ),
            "num_cudagraph_captured": compilation_counter.num_cudagraph_captured,
        }
        if samples:
            record["graphics_clock_mhz"] = samples[-1].graphics_clock_mhz
            record["mem_clock_mhz"] = samples[-1].mem_clock_mhz
        return record

    def _measure_idle(self, duration_s: float) -> dict[str, object]:
        samples = self.energy_meter.measure(duration_s)
        stats = self.energy_meter.summarize(samples)
        record = {
            "idle_duration_s": duration_s,
            "idle_avg_power_w": stats["avg_power_w"],
            "idle_energy_j": stats["energy_j"],
        }
        if samples:
            record["idle_graphics_clock_mhz"] = samples[-1].graphics_clock_mhz
            record["idle_mem_clock_mhz"] = samples[-1].mem_clock_mhz
        return record

    def _write_results(self, path: str, results: list[dict[str, object]]) -> None:
        if path.endswith(".csv"):
            self._write_csv(path, results)
        else:
            self._write_jsonl(path, results)

    def _write_jsonl(self, path: str, results: list[dict[str, object]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")

    def _write_csv(self, path: str, results: list[dict[str, object]]) -> None:
        if not results:
            return
        keys = sorted({k for row in results for k in row.keys()})
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    def _wrap_progress(self, iterable, desc: str):
        if not self.show_progress:
            return iterable
        try:
            from tqdm import tqdm  # type: ignore

            return tqdm(iterable, desc=desc)
        except Exception:
            return iterable
