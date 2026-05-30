# mini_vllm Profiling CLI

This folder contains a small profiling toolkit for `mini_vllm` that measures per-batch latency, power, and energy, and fits simple energy/latency models.

## Quick Start

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output profile.jsonl
```

## Requirements

- CUDA GPU and drivers.
- `nvidia-smi` available on `PATH` for power and clock sampling.
- Python env with `mini_vllm` dependencies.
- Optional: `pynvml` for more robust power sampling.
- Optional: `pyyaml` if you want YAML batch configs.

## CLI Options

Common flags:
- `--repeats 10`: repeat each batch to reduce noise.
- `--warmup 2`: warmup runs before measurement.
- `--idle_s 2.0`: measure idle power before batches.
- `--sample_interval_s 0.01`: power sampling interval.
- `--device_index 0`: GPU index.
- `--no_sync_cuda`: disable `torch.cuda.synchronize()` (not recommended for accurate timing).

Frequency / power control (optional):
- `--graphics_clock MIN,MAX`: fix graphics clock in MHz.
- `--power_limit_w W`: set GPU power limit in watts.

Modeling / plots (optional):
- `--model_out energy_latency_model.json`: save linear models.
- `--plot_prefix profile_plot`: generate scatter plots.

## Batch Config Format

```json
[
  {
    "name": "prefill_small",
    "type": "prefill",
    "requests": [
      {"context_len": 0, "query_len": 128},
      {"context_len": 0, "query_len": 256}
    ]
  },
  {
    "name": "decode_only",
    "type": "decode",
    "requests": [
      {"context_len": 128, "query_len": 1},
      {"context_len": 256, "query_len": 1}
    ]
  },
  {
    "name": "mixed",
    "type": "mixed",
    "requests": [
      {"context_len": 0, "query_len": 256},
      {"context_len": 512, "query_len": 1}
    ]
  }
]
```

Notes:
- `type` should be one of `prefill`, `decode`, or `mixed`.
- `context_len` + `query_len` determines the required KV blocks.

## Outputs

- `profile.jsonl` (or `.csv`): per-batch latency, average power, energy, plus idle baseline.
- `energy_latency_model.json`: linear models vs total tokens (if `--model_out` is set).
- `profile_plot_*.png`: scatter plots for latency and energy (if `--plot_prefix` is set).

## Example: Fixed GPU Clock

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output profile.jsonl \
  --graphics_clock 1200,1200
```

## Example: Model + Plots

```bash
python -m mini_vllm.profile.cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output profile.jsonl \
  --model_out energy_latency_model.json \
  --plot_prefix profile_plot
```

## CUDA Graph Phase Profiler

```bash
python -m mini_vllm.profile.cuda_graph_cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --output cuda_graph_profile.jsonl
```

This separate CLI runs each batch as:

1. warmup without CUDA graph
2. eager without CUDA graph, wrapped in a `cudaProfilerStart/Stop` capture region with NVTX labels
3. CUDA graph capture
4. CUDA graph replay, wrapped in a `cudaProfilerStart/Stop` capture region with NVTX labels

The output includes phase latencies plus a summary row with CUDA graph latency
gain and capture overhead fields.

## Automated CUDA Graph Metrics

```bash
python -m mini_vllm.profile.cuda_graph_nsys_cli \
  --model_name <your_model_name> \
  --batch_config path/to/batch_config.json \
  --cuda_visible_devices 0 \
  --device_index 0 \
  --output_prefix cuda_graph_metrics
```

This wrapper runs Nsight Systems, exports SQLite for each eager and graphed
capture range, and writes:

- `cuda_graph_metrics.phases.jsonl`: raw phase timings.
- `cuda_graph_metrics.summary.csv`: one row per batch.
- `cuda_graph_metrics.summary.jsonl`: the same merged rows in JSONL.

The summary includes eager time, capture time, graphed time, latency gain,
capture overhead, and sampled GPU metrics for both eager and graphed replay:
SM active, warp occupancy, DRAM bandwidth, and PCIe bandwidth. Unprefixed
metric columns use the graphed replay values.

## Omni pipeline experiments (vLLM-Omni)

These experiments target a **multi-stage** `vllm-omni` pipeline (thinker ->
talker -> code2wav) instead of the single-stream `mini_vllm` runner. Unlike
`cuda_graph_cli`, omni runs CUDA work in **separate worker processes**, so the
interesting signal here is genuine cross-stream concurrency, which the single
dense-LLM `cuda_graph_cli` will never show.

Install `vllm-omni` first (see the top-level README "Optional: vLLM-Omni"
section) and add the helper deps with `pip install -e ".[omni]"`.

A workload is a small JSON file (see `omni_workload_text.json` and
`omni_workload_mixed.json`) describing the model query type, output modalities,
prompt count, and token budget. The model is chosen at run time via `--model`.

### A) Latency + power, and (optionally) torch profiler

```bash
PYTHONPATH="$PWD/3rdparty/vllm:$PWD/3rdparty/vllm-omni:$PWD/src" \
python -m mini_vllm.profile.omni_profile_cli \
  --model Qwen/Qwen2.5-Omni-3B \
  --workload mini_vllm/profile/omni_workload_text.json \
  --output omni_profile.jsonl \
  --torch_profiler_dir omni_traces
```

`omni_profile.jsonl` contains an `omni_config` row, one `omni_stage_output` row
per streamed output (with `elapsed_ms` and `output_type`), and an `omni_summary`
row (wall time, power, energy). Setting `--torch_profiler_dir` exports
`VLLM_TORCH_PROFILER_DIR` and brackets generation with `start_profile` /
`stop_profile`, writing per-worker traces under that directory.

### B) Nsight Systems cross-stream concurrency + GPU metrics

```bash
PYTHONPATH="$PWD/3rdparty/vllm:$PWD/3rdparty/vllm-omni:$PWD/src" \
python -m mini_vllm.profile.omni_nsys_cli \
  --model Qwen/Qwen2.5-Omni-3B \
  --workload mini_vllm/profile/omni_workload_text.json \
  --cuda_visible_devices 0 \
  --device_index 0 \
  --output_prefix omni_metrics
```

This profiles the whole orchestrator+worker process tree (no
`--capture-range=cudaProfilerApi`, since that only annotates the orchestrator
which does no CUDA work), exports SQLite, and writes:

- `omni_metrics.phases.jsonl`: raw per-output timings.
- `omni_metrics.summary.csv` / `.summary.jsonl`: one row with concurrency
  metrics (`concurrent_ms`, `concurrency_ratio`, `max_concurrent_kernels`,
  `num_streams`, `num_pids`) plus sampled GPU metrics (SM active, warp
  occupancy, DRAM/PCIe bandwidth).

`concurrency_ratio` is the fraction of GPU-busy time during which two or more
kernels ran simultaneously. A single dense LLM forward stays near 0; a working
multi-stage omni pipeline should be meaningfully above 0.

### Summarize per-stage torch-profiler ops

```bash
python -m mini_vllm.profile.omni_torch_trace_summary \
  --trace_dir omni_traces \
  --out_jsonl omni_ops_summary.jsonl \
  --out_csv omni_ops_summary.csv \
  --top 25
```

This reduces the per-worker `ops_rank*.xlsx` artifacts into a flat per-stage op
table (top ops by self CUDA time), keyed by `stage` and `rank`.

Note on VRAM: `Qwen2.5-Omni-7B` keeps all stages resident and will not fit a
12 GB card; use a smaller `--model` (e.g. `Qwen2.5-Omni-3B`) or quantization.
